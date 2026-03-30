import logging
import torch
import os
import random as rd
import numpy as np

logger = logging.getLogger(__name__)

from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import DiceCELoss
from dataclasses import asdict

from ..config.config_params import  (
    DatasetConfig,
    ModelConfig,
    InferenceMetricsConfig,
    TrainingConfig    
)

from .data_manager import DafneDataModule
from .transform.transforms_builder import TransformBuilderTraining, TransformBuilderFineTuning
from ..utils.data_fingerprint import DatasetFingerprint
from ..models.factory import ModelFactory
from ..models.wrapper import DafneModelWrapper

from .engine.factory import create_supervised_trainer
from .engine.events import EngineEvents


def set_reproducibility(seed: int):
        '''
        Set all seeds for reproducibility
        '''
        import torch.backends.cudnn as cudnn
        rd.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


def callback_wrapper(callback):
    if hasattr(callback, 'emit'):
        return callback.emit
    elif callback is None: 
        return lambda *args: None
    else: 
        return callback

def run_training(model_config: ModelConfig,
                 dataset_config: DatasetConfig,
                 inference_metrics: InferenceMetricsConfig,
                 train_config: TrainingConfig,
                 save_path: str = None,
                 sig_status: callable = None, 
                 sig_progress: callable = None,
                 _callback_log: callable = None,
                 _callback_check_stop: callable = None,
                 sig_update_plot: callable = None,
                 sig_error: callable = None,
                 sig_stopped: callable = None,
                 sig_finished: callable = None,
                 device: str = 'cuda',
                 ):
    

    sig_status = callback_wrapper(sig_status)
    sig_progress = callback_wrapper(sig_progress)
    _callback_log = callback_wrapper(_callback_log)
    _callback_check_stop = callback_wrapper(_callback_check_stop)
    sig_update_plot = callback_wrapper(sig_update_plot)
    sig_error = callback_wrapper(sig_error)
    sig_stopped = callback_wrapper(sig_stopped)
    sig_finished = callback_wrapper(sig_finished)
    
    set_reproducibility(dataset_config.random_seed)

    model = None
    try:
        loaded_obj = None
        
        data_module = DafneDataModule(dataset_config, model_config, train_config)
        data_module.setup() # setup data file into train and validation
        data_list = data_module.data_list

        sig_status(f"Training initialization on: {device} device")
        sig_status(f"Dataset loading ({len(data_list)} files...)")

        data_fingerprint = DatasetFingerprint(data_module.train_files, spatial_dims=model_config.spatial_dims)

        if model_config.fine_tuning:
            sig_status("Fine-tuning mode")

            from dafne_dl.DynamicTorchModel import DynamicTorchModel
            with open(train_config.pretrained_model_path, "rb") as f:
                loaded_obj = DynamicTorchModel.Load(f)
            
            net_params = loaded_obj.metadata['net_metadata']
            model_config.model_name = net_params.get('model_name', model_config.model_name)
            model_config.spatial_dims = net_params.get('spatial_dims')
            model_config.in_channels = net_params.get('in_channels')
            model_config.use_dynamic = net_params.get('use_dynamic')
            model_config.patch_size = net_params.get('patch_size')
            model_config.median_shape = net_params.get('median_shape')
            model_config.median_spacing = net_params.get('median_spacing')
            model_config.n_levels = net_params.get('n_levels', None)
            model_config.labels_name = net_params.get('labels_name', None)
            model_config.extra_params['n_levels'] = net_params.get('n_levels', None)
            model_config.extra_params['kernel_size'] = net_params.get('kernel_size', None)
            model_config.extra_params['kernels'] = net_params.get('kernels', None)
            model_config.extra_params['strides'] = net_params.get('strides', None)
        
            data_builder = TransformBuilderFineTuning(model_config, train_config)
        else:
            sig_status("Training from scratch mode")
            model_config.median_shape = data_fingerprint.data_shape.tolist()
            model_config.median_spacing = data_fingerprint.data_spacing.tolist()
            data_builder = TransformBuilderTraining(model_config, train_config, data_fingerprint)
        train_transforms, val_transforms = data_builder.build_transforms()
        
        data_module.create_datasets(train_transforms, val_transforms) # create datasets
        train_loader, val_loader = data_module.create_dataloaders() # create dataloaders

        out_channels = data_fingerprint.count_label_mask(data_module.train_files) #compute n_classes on train files
        model_config.out_channels = out_channels
        labels_name = [s.replace('mask_', '') for s in data_fingerprint.get_labels_name()] # mantain only labels name without 'mask' substring
        model_config.labels_name = labels_name
        
        use_gradual_unfreezing = model_config.gradual_unfreezing and \
            model_config.fine_tuning 

        core_model = ModelFactory.create_model(model_config, loaded_obj)
        unfreeze_fn = ModelFactory.gradual_unfreeze if use_gradual_unfreezing else None
        model = DafneModelWrapper(core_model)
        model.to(device)

        if loaded_obj is not None:
            import gc
            loaded_obj = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if model_config.fine_tuning and \
            not model_config.lora_config and \
            not (model_config.percent_to_freeze and model_config.percent_to_freeze > 0):
            # Full fine-tuning (no freeze, no LoRA): discriminative LR,
            # lower for earlier layers and higher for later layers.
            param_groups = ModelFactory.change_lr_for_layer(model.model, train_config.learning_rate)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        else:
            # Freeze mode or LoRA: uniform LR on trainable params only.
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=train_config.learning_rate, 
                weight_decay=1e-4
            )

        criterion = DiceCELoss(include_background=False,
                                softmax=True,
                                to_onehot_y=True,
                                squared_pred=True,
                                smooth_dr=1e-5)
            
        scheduler = None
        if train_config.scheduler:
            scheduler = CosineAnnealingLR(optimizer, T_max=train_config.epochs, eta_min=1e-6)

        trainer = create_supervised_trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            spatial_dims=model_config.spatial_dims,
            val_roi_size=model_config.patch_size,
            mixed_precision=train_config.mixed_precision,
            val_loader=val_loader,
            params={'inference_metrics': asdict(inference_metrics)},
            labels_name=model_config.labels_name,
            save_path=save_path,
            model_name=model_config.model_name,
            scheduler=scheduler,
            sig_update_plot=sig_update_plot,
            unfreeze_fn=unfreeze_fn,
            early_stopping=train_config.early_stopping,
            initial_freeze_degree=model_config.percent_to_freeze \
                if model_config.percent_to_freeze is not None else 0.0,
            on_log=_callback_log
        )

        def on_epoch_progress(engine):
            epoch = engine.state.epoch
            total = engine.state.max_epochs
            loss = engine.state.metrics.get('avg_loss', 0.0)
            sig_progress(int((epoch / total) * 100))
            sig_status(f"Epoch {epoch}/{total} | Loss: {loss:.3f}")

        trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, on_epoch_progress)

        trainer.run(train_loader,
                    max_epochs=train_config.epochs,
                    check_stop=_callback_check_stop)

        save_dir = os.path.dirname(save_path)
        model_filename = os.path.basename(save_path).split('.')[0]
        best_weights_path = os.path.join(save_dir, f"{model_filename}_best_model.pth")
        if os.path.exists(best_weights_path):
            best_weights = torch.load(best_weights_path, map_location='cpu')
            model.load_weights(best_weights)

        sig_status("Packaging the model into .model format...")
        try:
            model.save_model_and_metadata(model_config,
                                            train_config,
                                            save_path)
            sig_status("Model packaged successfully")
        except Exception as e:
            logger.error("Model packaging failed: %s", e, exc_info=True)
            sig_error(str(e))

        if os.path.exists(best_weights_path):
            os.remove(best_weights_path)

        import gc
        del optimizer
        criterion = None
        scheduler = None
        if model is not None:
            model.to('cpu')
            model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if _callback_check_stop and _callback_check_stop():
            sig_stopped()
        else:
            sig_finished()

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        sig_error(str(e))
    finally:
        import gc
        if model is not None:
            try:
                model.to('cpu')
            except Exception:
                pass
            model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()