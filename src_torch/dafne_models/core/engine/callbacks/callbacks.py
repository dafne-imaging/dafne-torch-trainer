import logging
import torch
import os
import numpy as np

from .base_callback import BaseCallback

logger = logging.getLogger(__name__)
from typing import Dict, Any

from monai.transforms import Compose, AsDiscrete
from monai.metrics import  (DiceMetric, 
                            HausdorffDistanceMetric, 
                            SurfaceDistanceMetric, 
                            ConfusionMatrixMetric,
                            compute_confusion_matrix_metric)
import random as rd
from monai.data import decollate_batch

MONAI_REGISTRY = {
    'dice': DiceMetric,
    'hausdorff_95': HausdorffDistanceMetric,
    'surface_distance': SurfaceDistanceMetric,
    'precision': ConfusionMatrixMetric,
    'sensitivity': ConfusionMatrixMetric,
    'threat_score': ConfusionMatrixMetric,
    'specificity': ConfusionMatrixMetric,
    'accuracy': ConfusionMatrixMetric,
    'f1_score': ConfusionMatrixMetric,
}

class MetricsCallback(BaseCallback):
    '''
    Metrics callback.
    '''
    def __init__(self, params: Dict[str, Any], n_classes:int, labels_name:list[str]=None):
        '''
        Initialize the metrics callback.
        Args: 
            params: Dictionary of parameters from .
            n_classes: Number of classes.
            labels_name: List of labels names.
        '''
        super().__init__()
        self.params = params
        self.n_classes = n_classes
        self.labels_name = labels_name

        self.active_metrics = self._build_active_metrics()

        # post processing transforms for validation
        self.post_pred = Compose([
            AsDiscrete(argmax=True, to_onehot=self.n_classes)
        ])
        self.post_label = Compose([
            AsDiscrete(to_onehot=self.n_classes)
        ])
    
    def _build_active_metrics(self):
        active_metrics = {}
        for key, value in self.params['inference_metrics'].items():
            if key.startswith('compute_') and value:
                metric = key.replace('compute_', '')
                if metric in MONAI_REGISTRY:
                    if issubclass(MONAI_REGISTRY[metric], ConfusionMatrixMetric):
                        metric_monai_name = metric.replace('_', ' ')
                        active_metrics[metric] = MONAI_REGISTRY[metric](
                            include_background=self.params['inference_metrics']['include_background'], 
                            reduction=self.params['inference_metrics']['reduction'],
                            metric_name=metric_monai_name
                        )
                    else:
                        active_metrics[metric] = MONAI_REGISTRY[metric](
                            include_background=self.params['inference_metrics']['include_background'], 
                            reduction=self.params['inference_metrics']['reduction']
                        )
        return active_metrics
    
    def on_iteration_completed(self, engine):
        val_output = engine.state.output['y_pred']
        val_mask = engine.state.output['y']

        val_output_decollate = decollate_batch(val_output)
        val_label_decollate = decollate_batch(val_mask)
        val_preds = [self.post_pred(i) for i in val_output_decollate]
        val_masks = [self.post_label(i) for i in val_label_decollate]

        for metric in self.active_metrics.values():
            metric(y_pred=val_preds, y=val_masks)

    def on_epoch_completed(self, engine):
        for metric_name, metric in self.active_metrics.items():
            metric_score = metric.aggregate()
            
            # In case metric.aggregate() returns a list across multiple batches, concatenate them
            if isinstance(metric_score, list):
                metric_score = torch.cat(metric_score, dim=0)

            raw_buffer = metric.get_buffer()
            if isinstance(metric, ConfusionMatrixMetric):
                computed_buffer = compute_confusion_matrix_metric(metric_name.replace('_', ' '), raw_buffer)
            else:
                computed_buffer = raw_buffer
            if self.params['inference_metrics']['include_background']:
                computed_buffer = computed_buffer[:, 1:]
            engine.state.metrics[f'buffer_{metric_name}'] = computed_buffer
            active_scores = metric_score[1:] if self.params['inference_metrics']['include_background'] \
                else metric_score
            metric_score_avg = active_scores.mean().item()
            per_mask_metric_score = {name: active_scores[i].item() for i, name in enumerate(self.labels_name) if i < len(active_scores)}
            engine.state.metrics[f'avg_{metric_name}'] = metric_score_avg
            for label in self.labels_name:
                if label in per_mask_metric_score:
                    engine.state.metrics[f'{metric_name}_{label}'] = per_mask_metric_score[label]
            metric.reset()
    

class VisualizationCallback(BaseCallback):
    def __init__(self, sig_update_plot, dataloader):
        super().__init__()
        self.sig_update_plot = sig_update_plot
        self.dataloader = dataloader
    
    def on_epoch_completed(self, engine):
        dataset = self.dataloader.dataset
        idx = rd.randrange(len(dataset))
        sample = dataset[idx]
        
        current_spacing = sample['image_meta_dict']['pixdim']
        if isinstance(current_spacing, torch.Tensor):
            current_spacing = tuple(current_spacing.cpu().numpy())
        img_spacing = current_spacing[1:]
        
        engine.state.pred_dice_per_mask = engine.process_function.__self__.compute_predictions(engine, sample)
        
        avg_loss = engine.state.metrics.get('avg_loss', 0.0)
        val_loss = engine.state.metrics.get('avg_val_loss', 0.0)
        best_dice = engine.state.best_metric if engine.state.best_metric is not None else -float('inf')
        vis_data = engine.state.pred_dice_per_mask
        per_mask_dice_score = {k: v for k, v in engine.state.metrics.items() if k.startswith('dice_')}
        
        self.sig_update_plot(
            avg_loss,
            vis_data['image'],
            vis_data['mask'],
            val_loss,
            img_spacing,
            best_dice,
            per_mask_dice_score
        )

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path:str, model_name:str, monitor:str='avg_dice', on_log=None):
        super().__init__()
        self.save_path = save_path
        self.model_name = model_name
        self.monitor = monitor
        self.on_log = on_log

    def on_epoch_completed(self, engine):
        current_val = engine.state.metrics.get(self.monitor, None)
        if current_val is None:
            return

        if engine.state.best_metric is None:
            engine.state.best_metric = -float('inf')

        if current_val > engine.state.best_metric:
            engine.state.best_metric = current_val

            if self.save_path:
                save_dir = os.path.dirname(self.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                model_filename = os.path.basename(self.save_path).split('.')[0]
                filename = f"{model_filename}_best_model.pth"
                best_model_path = os.path.join(save_dir, filename)
                torch.save(engine.process_function.__self__.model.state_dict(), best_model_path)
                msg = f'New best {self.monitor}: {current_val:.4f}. Model saved!'
                if self.on_log:
                    self.on_log(msg)
                else:
                    logger.info(msg)


class GradualUnfreezeCallback(BaseCallback): 
    def __init__(self, initial_freeze_degree, unfreeze_fn, task, on_log):
        super().__init__()
        self.unfreeze_fn = unfreeze_fn
        self.task = task
        self.initial_freeze_degree = initial_freeze_degree
        self.on_log = on_log
    
    def on_epoch_started(self, engine):
        self._gradual_unfreeze(engine)

    def _gradual_unfreeze(self, engine):
        epoch = engine.state.epoch
        interval = max(1, int(engine.state.max_epochs * 0.10))
        if epoch > 0 and epoch % interval == 0:
            start_trainable_percent = 1.0 - self.initial_freeze_degree
            unfreeze_progress = epoch / max(1, int(engine.state.max_epochs * 0.75))
            current_trainable_degree = min(1.0, start_trainable_percent + unfreeze_progress)
            self.task.optimizer = self.unfreeze_fn(self.task.model.model, current_trainable_degree, self.task.optimizer)
            msg = f"Gradual Unfreeze: now {current_trainable_degree*100:.1f}% of the network is trainable"
            if self.on_log:
                self.on_log(msg)
            else:
                logger.info(msg)


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience:int=20, monitor:str='avg_dice', on_log=None):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.wait_count = 0
        self._best = None
        self.on_log = on_log

    def on_epoch_completed(self, engine):
        current_metric = engine.state.metrics.get(self.monitor, None)
        if current_metric is None:
            return
        if self._best is None or current_metric > self._best:
            self._best = current_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                engine.state.terminated = True
                msg = f'Training interrupted because of early stopping after {self.patience} epochs without improvement.'
                if self.on_log:
                    self.on_log(msg)
                else:
                    logger.info(msg)


class ClearGPUMemory(BaseCallback):
    '''
    Frees GPU tensors stored in engine state after each validation batch,
    then runs gc.collect() + empty_cache() at the end of every epoch.
    Must be registered on the evaluator AFTER MetricsCallback.
    '''
    def on_iteration_completed(self, engine):
        if isinstance(engine.state.output, dict):
            engine.state.output.pop('y_pred', None)
            engine.state.output.pop('y', None)
        engine.state.batch = None
        engine.state.output = None

    def on_epoch_completed(self, engine):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ContinualLearningCallback(BaseCallback): 
    '''
        Save matrix theta_{A_i}* at the end of original training 
    '''

    def __init__(self, val_loader, device, save_path, lambda_reg, criterion,
                 spatial_dims=2, val_roi_size=None):
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.lambda_reg = lambda_reg
        self.spatial_dims = spatial_dims
        self.val_roi_size = val_roi_size

    def on_completed(self, engine):
        from monai.inferers import sliding_window_inference
        self.model = engine.process_function.__self__.model
        self.model.eval()

        fisher = {name:torch.zeros_like(param) for name, param in \
                  self.model.named_parameters() if param.requires_grad}

        params_snapshot = {name: param.clone() for name, param in \
                        self.model.state_dict().items()}

        n_batches = 0
        for batch in self.val_loader:
            inputs = batch['image'].to(self.device)
            target = batch['mask'].long().to(self.device)

            self.model.zero_grad()

            if self.spatial_dims == 3 and self.val_roi_size is not None:
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=self.val_roi_size,
                    sw_batch_size=4,
                    overlap=0.25,
                    predictor=self.model
                )
            else:
                outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None: 
                    fisher[name] += param.grad.detach() ** 2

            n_batches += 1

        for name in fisher: 
            fisher[name] /= n_batches
        
        model_name = os.path.basename(self.save_path).split('.')[0]
        ewc_path = os.path.join(os.path.dirname(self.save_path), f'{model_name}_ewc.pt')
        torch.save({
            'fisher': fisher,
            'params_snapshot': params_snapshot
        }, ewc_path)
        ewc_cpu = {
            'fisher': {k: v.cpu().numpy() for k, v in fisher.items()},
            'params_snapshot': {k: v.cpu().numpy() for k, v in params_snapshot.items()}
        }
        self.model.ewc_data = ewc_cpu