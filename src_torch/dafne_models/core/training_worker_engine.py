import os
import traceback
import random as rd
import numpy as np
from dataclasses import asdict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.data.utils import pad_list_data_collate

from PyQt5.QtCore import QThread, pyqtSignal

from ..utils.data_fingerprint import DatasetFingerprint

from .engine.factory import create_supervised_trainer
from .engine.events import EngineEvents

from ..config.config_params import ModelConfig, DatasetConfig, TrainingConfig, InferenceMetricsConfig
from ..models.factory import ModelFactory
from ..models.wrapper import DafneModelWrapper
from .data_manager import DafneDataModule
from .transform.transforms_builder import TransformBuilderTraining, TransformBuilderFineTuning


class TrainingWorker(QThread):

    '''
        Worker class that runs the PyTorch training loop in a separate thread.
        Ensures the GUI remains responsive during intensive computation.
        Training Worker save model training parameters and memory buffer from original dataset
        for future incremental learning in fine-tuning
    '''
    
    # Send data to cpu (float, numpy, numpy, float, numpy, float)
    sig_update_plot = pyqtSignal(int, float, object, object, float, object, float, dict)

    # Send status information for user console
    sig_status = pyqtSignal(str)

    # Send error information to user console
    sig_error = pyqtSignal(str)

    # progress values to sent to GUI
    sig_progress = pyqtSignal(int)

    # Send message when training ended
    sig_finished = pyqtSignal()

    # stopped signal
    sig_stopped = pyqtSignal()

    def __init__(self,
                 dataset_config:DatasetConfig,
                 model_config:ModelConfig,
                 train_config:TrainingConfig, 
                 inference_metrics:InferenceMetricsConfig,
                 save_path:str=None
                 ):
        
        super().__init__()
        
        # inzialize worker parameters
        self.is_running = True

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.train_config = train_config
        self.inference_metrics = inference_metrics
        self.save_path = save_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _callback_check_stop(self):
        '''
        Check if user stop the training
        
        :param self: Descrizione
        '''
        return not self.is_running

    def _callback_log(self, message):
        '''
        Callback function for Engine logs
        
        :param self
        :param message: log description
        '''
        self.sig_status.emit(message)
    
    @staticmethod
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
        # Optional: ensure that even dataloader workers are seeded (if used)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def run(self):
        '''
        Run training loop
        '''
        # Set reproducibility at the very beginning of the thread execution
        self.set_reproducibility(self.dataset_config.random_seed)
        
        try:
            loaded_obj = None
            
            data_module = DafneDataModule(self.dataset_config, self.model_config, self.train_config)
            data_module.setup() # setup data file into train and validation
            data_list = data_module.data_list

            self.sig_status.emit(f"Training initialization on: {self.device} device")
            self.sig_status.emit(f"Dataset loading ({len(data_list)} files...)")

            data_fingerprint = DatasetFingerprint(data_module.train_files, spatial_dims=self.model_config.spatial_dims)
            self.model_config.labels_name = data_fingerprint.get_labels_name()
            
            if self.model_config.fine_tuning:
                self.sig_status.emit("Fine-tuning mode")

                from dafne_dl.DynamicTorchModel import DynamicTorchModel
                with open(self.train_config.pretrained_model_path, "rb") as f:
                    loaded_obj = DynamicTorchModel.Load(f)
                
                net_params = loaded_obj.metadata['net_metadata']
                self.model_config.model_name = net_params.get('model_name', self.model_config.model_name)
                self.model_config.spatial_dims = net_params.get('spatial_dims')
                self.model_config.in_channels = net_params.get('in_channels')
                self.model_config.use_dynamic = net_params.get('use_dynamic')
                self.model_config.patch_size = net_params.get('patch_size')
                self.model_config.median_shape = net_params.get('median_shape')
                self.model_config.median_spacing = net_params.get('median_spacing')
                self.model_config.n_levels = net_params.get('n_levels', None)
                self.model_config.labels_name = net_params.get('labels_name', None)
                self.model_config.extra_params['n_levels'] = net_params.get('n_levels', None)
                self.model_config.extra_params['kernel_size'] = net_params.get('kernel_size', None)
                self.model_config.extra_params['kernels'] = net_params.get('kernels', None)
                self.model_config.extra_params['strides'] = net_params.get('strides', None)
            
                data_builder = TransformBuilderFineTuning(self.model_config, self.train_config)
            else:
                self.sig_status.emit("Training from scratch mode")
                self.model_config.median_shape = data_fingerprint.data_shape.tolist()
                self.model_config.median_spacing = data_fingerprint.data_spacing.tolist()
                data_builder = TransformBuilderTraining(self.model_config, self.train_config, data_fingerprint)
            train_transforms, val_transforms = data_builder.build_transforms()
            
            data_module.create_datasets(train_transforms, val_transforms) # create datasets
            train_loader, val_loader = data_module.create_dataloaders() # create dataloaders

            out_channels = data_fingerprint.count_label_mask(data_module.train_files) #compute n_classes on train files
            self.model_config.out_channels = out_channels
            self.model_config.labels_name = data_fingerprint.get_labels_name()
            
            use_gradual_unfreezing = self.model_config.gradual_unfreezing and \
                self.model_config.fine_tuning 

            core_model = ModelFactory.create_model(self.model_config, loaded_obj)
            unfreeze_fn = ModelFactory.gradual_unfreeze if use_gradual_unfreezing else None
            self.model = DafneModelWrapper(core_model)
            self.model.to(self.device)

            if loaded_obj is not None:
                import gc
                loaded_obj = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.model_config.fine_tuning and \
                not self.model_config.lora_config and \
                not (self.model_config.percent_to_freeze and self.model_config.percent_to_freeze > 0):
                # Full fine-tuning (no freeze, no LoRA): discriminative LR,
                # lower for earlier layers and higher for later layers.
                param_groups = ModelFactory.change_lr_for_layer(self.model.model, self.train_config.learning_rate)
                optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
            else:
                # Freeze mode or LoRA: uniform LR on trainable params only.
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.train_config.learning_rate, 
                    weight_decay=1e-4
                )

            criterion = DiceCELoss(include_background=False,
                                    softmax=True,
                                    to_onehot_y=True,
                                    squared_pred=True,
                                    smooth_dr=1e-5)
                
            scheduler = None
            if self.train_config.scheduler:
                scheduler = CosineAnnealingLR(optimizer, T_max=self.train_config.epochs, eta_min=1e-6)

            trainer = create_supervised_trainer(
                model=self.model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                spatial_dims=self.model_config.spatial_dims,
                val_roi_size=self.model_config.patch_size,
                mixed_precision=self.train_config.mixed_precision,
                val_loader=val_loader,
                params={'inference_metrics': asdict(self.inference_metrics)},
                labels_name=self.model_config.labels_name,
                save_path=self.save_path,
                model_name=self.model_config.model_name,
                scheduler=scheduler,
                sig_update_plot=self.sig_update_plot,
                unfreeze_fn=unfreeze_fn,
                early_stopping=self.train_config.early_stopping,
                initial_freeze_degree=self.model_config.percent_to_freeze \
                    if self.model_config.percent_to_freeze is not None else 0.0,
                on_log=self._callback_log
            )

            def on_epoch_progress(engine):
                epoch = engine.state.epoch
                total = engine.state.max_epochs
                loss = engine.state.metrics.get('avg_loss', 0.0)
                self.sig_progress.emit(int((epoch / total) * 100))
                self.sig_status.emit(f"Epoch {epoch}/{total} | Loss: {loss:.3f}")

            trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, on_epoch_progress)

            trainer.run(train_loader,
                        max_epochs=self.train_config.epochs,
                        check_stop=self._callback_check_stop)

            save_dir = os.path.dirname(self.save_path)
            model_filename = os.path.basename(self.save_path).split('.')[0]
            best_weights_path = os.path.join(save_dir, f"{model_filename}_best_model.pth")
            best_weights = torch.load(best_weights_path, map_location='cpu')
            self.model.load_weights(best_weights)
            
            self.sig_status.emit("Packaging the model into .dafne format...")
            try:
                self.model.save_model_and_metadata(self.model_config, 
                                                self.train_config, 
                                                self.save_path)
                self.sig_status.emit("Model packaged successfully")
            except Exception as e:
                traceback.print_exc()
                self.sig_error.emit(str(e))

            if os.path.exists(best_weights_path):
                os.remove(best_weights_path)

            import gc
            del optimizer
            criterion = None
            scheduler = None
            if hasattr(self, 'model') and self.model is not None:
                self.model.to('cpu')
                self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not self.is_running:
                self.sig_stopped.emit()
            else:
                self.sig_finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.sig_error.emit(str(e))
        finally:
            import gc
            if hasattr(self, 'model') and self.model is not None:
                try:
                    self.model.to('cpu')
                except Exception:
                    pass
                self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stop(self):
        self.is_running = False            