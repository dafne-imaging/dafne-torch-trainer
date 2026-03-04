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

from .pytorch_loop import pytorch_training_loop
from ..utils.data_fingerprint import DatasetFingerprint

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
    sig_update_plot = pyqtSignal(float, object, object, float, object, float, dict)

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

    def _callback_epoch_end(self, epoch, loss, img, mask, val_loss, spacing, best_dice, per_mask_dice_score):
        
        '''
        Function called at the end of each epoch by the Engine
        '''

        self.sig_update_plot.emit(loss, img, mask, val_loss, spacing, best_dice, per_mask_dice_score)
        total_epochs = self.train_config.epochs
        current_epoch = epoch + 1
        percent = int((current_epoch / total_epochs) * 100)

        self.sig_progress.emit(percent) # -> send epochs percent to progress bar
        self.sig_status.emit(f"Epoch {current_epoch}/{total_epochs} | Loss: {loss:.3f}")
    
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
    
    def run(self):
        '''
        Run training loop
        '''
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

            core_model = ModelFactory.create_model(self.model_config, loaded_obj)
            self.model = DafneModelWrapper(core_model)
            self.model.to(self.device)

            if loaded_obj is not None:
                import gc
                loaded_obj = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.model_config.fine_tuning and \
                not self.model_config.lora_config:
                param_groups = ModelFactory.change_lr_for_layer(self.model.model, self.train_config.learning_rate)
                optimizer = torch.optim.Adam(param_groups)
            else:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 
                    lr=self.train_config.learning_rate
                )

            criterion = DiceCELoss(include_background=False, 
                                    softmax=True,
                                    to_onehot_y=True,
                                    squared_pred=True, 
                                    smooth_dr=1e-5) # for nan loss
                
            scheduler = None
            if self.train_config.scheduler:
                scheduler = CosineAnnealingLR(optimizer, T_max=self.train_config.epochs, eta_min=1e-6)

            best_val_dice = pytorch_training_loop(
                model=self.model,
                optimizer=optimizer,
                train_dataloader = train_loader,
                valid_dataloader = val_loader,
                criterion=criterion,
                device=self.device,
                epochs=self.train_config.epochs,
                #callback injection
                save_path=self.save_path,
                scheduler=scheduler,
                on_epoch_end=self._callback_epoch_end,
                check_stop=self._callback_check_stop,
                on_log=self._callback_log,
                early_stopping=self.train_config.early_stopping,
                n_classes=out_channels,
                mixed_precision=self.train_config.mixed_precision,
                spatial_dims=self.model_config.spatial_dims,
                val_roi_size=self.model_config.patch_size,
                model_name=self.model_config.model_name,
                labels_name=self.model_config.labels_name,
                inference_metrics=asdict(self.inference_metrics)
            )

            save_dir = os.path.dirname(self.save_path)
            best_weights_path = os.path.join(save_dir, f"{self.model_config.model_name}_best_model.pth")
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
            
            if torch.cuda.is_available():
                self.model.to('cpu')
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            if not self.is_running:
                self.sig_stopped.emit()
            else:
                self.sig_finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.sig_error.emit(str(e))

    def stop(self):
        self.is_running = False            