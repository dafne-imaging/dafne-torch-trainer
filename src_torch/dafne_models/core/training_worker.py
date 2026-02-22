import os
import json
import traceback
import random as rd
import numpy as np

import torch

from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.data.utils import pad_list_data_collate

from PyQt5.QtCore import QThread, pyqtSignal

from sklearn.model_selection import train_test_split

from dafne_dl.DynamicTorchModel import DynamicTorchModel
from .lora.lora_models import LoRAModel
from .pytorch_loop import pytorch_training_loop
from .utils import count_label_mask, get_median_spacing
from ..utils.data_fingerprint import DatasetFingerprint
from ..utils.optimizer import get_optimal_hyperparameters
from ..models.dafne_networks import DafneUnetModel, DafneDynUnet
from ..bin.create_torch_model import create_dynamic_model
from .transforms_builder import (build_transform_list, 
                                 build_transforms_dynunet)



class TrainingWorker(QThread):

    '''
        Worker class that runs the PyTorch training loop in a separate thread.
        Ensures the GUI remains responsive during intensive computation.
        Training Worker save model training parameters and memory buffer from original dataset
        for future incremental learning in fine-tuning
    '''
    
    # Send data to cpu (float, numpy, numpy)
    sig_update_plot = pyqtSignal(float, object, object, float, object)

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
                 file_list:list,
                 model_params:dict,
                 train_params:dict, 
                 pretrained_model_path:str=None, #.dafne file path
                 save_path:str=None,
                 early_stopping:bool=False,
                 dyn_model_params:dict=None,
                 adaptation_params:dict=None,
                 ):
        
        super().__init__()
        
        # inzialize worker parameters
        self.file_list = file_list
        self.model_params = model_params
        self.dyn_unet_params = dyn_model_params
        self.train_params = train_params
        self.save_path = save_path
        self.adaptation_params = adaptation_params

        self.is_running = True
        self.early_stopping = early_stopping

        self.pretrained_model_path = pretrained_model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _callback_epoch_end(self, epoch, loss, img, mask, val_loss, spacing):
        
        '''
        Function called at the end of each epoch by the Engine
        '''

        self.sig_update_plot.emit(loss, img, mask, val_loss, spacing)
        total_epochs = self.train_params.get('epochs')
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
    
    def _save_model(self):
        return

    def _save_model_params(self, params):
        '''
        Save model params in json file
        '''
    
    def _setup_model(self):
        '''
        Setup model for training model from scratch
        
        :param self
        '''
        
        if self.model_params['model_name'] == 'unet':
            return DafneUnetModel(spatial_dims=self.model_params.get('spatial_dims', 2),
                                n_levels=self.model_params.get('n_levels', 5),
                                kernel_size=self.model_params.get('kernel_size', 3),
                                out_channels=self.model_params['n_classes'],
                                in_channels=self.model_params.get('in_channels', 1))
        elif self.model_params['model_name'] == 'dynunet':
            return DafneDynUnet(spatial_dims = self.model_params.get('spatial_dims', 3),
                                in_channels=self.model_params.get('in_channels', 1),
                                out_channels=self.model_params['n_classes'],
                                kernel_size=self.model_params.get('kernels'),
                                strides=self.model_params.get('strides'),
                                norm_name=("INSTANCE", {"affine": True}),
                                deep_supervision=False)
        else:
            raise ValueError(f'Model {self.model_params["model_name"]} not found')

    def _setup_training(self):
        '''
        Setup training parameters
        It depends on the networks parameters
        
        :param n_classes: number of classes
        '''
        spatial_dims = self.model_params.get('spatial_dims', 3)
        use_dynamic = self.model_params.get('use_dynamic', False)
        augm_params = self.train_params.get('augmentation', {})
        batch_size = self.train_params.get('batch_size', 2)

        fingerprint = DatasetFingerprint(self.file_list, spatial_dims=spatial_dims)
        median_spacing = fingerprint.data_spacing
        median_shape = fingerprint.data_shape
       
        self.model_params['kernels'] = None
        self.model_params['strides'] = None
        self.model_params['data_shape'] = median_shape
        self.model_params['median_spacing'] = median_spacing
        self.model_params['spatial_dims'] = spatial_dims
        self.model_params['in_channels'] = self.model_params.get('in_channels', 1)
        self.model_params['out_channels'] = self.model_params.get('n_classes', 2)

        # dynamic mode
        if use_dynamic and spatial_dims == 3:
            self.sig_status.emit("Dynamic Mode: Analyzing Dataset Fingerprint...")

            patch_size, batch_size = get_optimal_hyperparameters(
                median_shape, spatial_dims=spatial_dims
            )
            final_patch_size = patch_size
            
            self.sig_status.emit(f"Auto-Config: Patch={patch_size}, Batch={batch_size}")
            
            model_name = 'dynunet'
            self.model_params['model_name'] = model_name
            kernels, strides = fingerprint.get_kernel_and_strides(patch_size)

            self.model_params['kernels'] = kernels
            self.model_params['strides'] = strides

            model = self._setup_model().to(device=self.device)
            
            train_transforms =  build_transforms_dynunet(
                keys=['image', 'mask'],
                patch_size=patch_size,
                target_spacing=median_spacing,
                train_transforms=True,
                augm_params=augm_params
            )

            valid_transforms = build_transforms_dynunet(
                keys=['image', 'mask'],
                patch_size=patch_size,
                target_spacing=median_spacing,
                train_transforms=False, 
                augm_params=augm_params
            )

        else:
            self.sig_status.emit("Classic Mode: Using Standard U-Net")
            
            final_patch_size = (16, 96, 96)

            model_name = 'unet'
            self.model_params['model_name'] = model_name
            self.model_params['data_shape'] = median_shape
            
            model = self._setup_model().to(device=self.device)
            
            train_transforms = build_transform_list(['filepath'],
                                                median_spacing,
                                                True,
                                                augm_params,
                                                self.model_params['spatial_dims'])

            valid_transforms = build_transform_list(['filepath'],
                                                    median_spacing,
                                                    False,
                                                    augm_params,
                                                    self.model_params['spatial_dims'])
        
        self.model_params['batch_size'] = batch_size
        self.model_params['patch_size'] = final_patch_size
        
        return model, train_transforms, valid_transforms


    def _save_model_params(self):
        # here save params dictionary to json file    
        return
    
    def _setup_fine_tuning(self, 
                            dyn_torch_model_obj:DynamicTorchModel,
                            percent_to_freeze:float,
                            lora_config:dict,
                            n_classes:int,
                            ) -> None:
        '''
        Setup fine tuning parameters
        It depends on the networks parameters
        
        :param dynamic_torch_model: dynamic torch model from which retrive model and metadata
        :param n_classes: number of classes for the actual model
        :param percent_to_freeze: percent of layers to freeze
        '''
        try:
            model = dyn_torch_model_obj.model # get model object (ex. dynunet, unet, etc.)
            net_params = dyn_torch_model_obj.metadata['net_metadata'] # get net params
        except Exception as e:
            raise ValueError(f'Error loading model or pretrained weights: {e}')

        # update output channels if needed
        original_n_classes = net_params.get('out_channels', None) #read original model number of classes
        if original_n_classes != n_classes:
            model.update_output_channels(n_classes)
            model.to(self.device)
            self.model_params['out_channels'] = n_classes
            self.sig_status.emit(f"Updated model output channels from {original_n_classes} to {n_classes}")
        
        if lora_config is not None:
            if not isinstance(lora_config, dict):
                raise ValueError('Invalid type! The lora_config should be a dictionary')
            
            model = LoRAModel(model, lora_config)
            model.enable_adapter() #enable lora adapters
            model.to(self.device)
    
        if percent_to_freeze is not None and lora_config is None:
            self.freeze_layers(model, percent_to_freeze) #freeze layers and de-freeze norm and bn layers
        
        final_patch_size = net_params.get('patch_size', (16, 96, 96)) #read original model patch size
        self.model_params['patch_size'] = final_patch_size #update model params

        model_name = net_params.get('model_name', 'unet') #read original model name
        self.model_params['model_name'] = model_name #update model params

        median_spacing = net_params.get('median_spacing', (1.0, 1.0, 1.0)) #read original model spacing
        self.model_params['median_spacing'] = median_spacing #update model params

        median_shape = net_params.get('data_shape', None) #read original model shape
        self.model_params['data_shape'] = median_shape #update model params

        augm_params = self.train_params.get('augmentation', {}) #read augmentation parameters
        batch_size = self.train_params.get('batch_size', 2) #read batch size
        
        kernels = net_params.get('kernels', None) #read original model kernels
        strides = net_params.get('strides', None) #read original model strides
        self.model_params['kernels'] = kernels #update model kernels
        self.model_params['strides'] = strides #update model strides
        
        spatial_dims = net_params.get('spatial_dims', 2) #read spatial dimensions
        self.model_params['spatial_dims'] = spatial_dims #update model spatial dims
        
        if net_params.get('use_dynamic'):
            patch_size, auto_batch_size = get_optimal_hyperparameters(
                    median_shape, spatial_dims=spatial_dims
                )
            final_patch_size = patch_size

            train_transforms =  build_transforms_dynunet(
                    keys=['image', 'mask'],
                    patch_size=patch_size,
                    target_spacing=median_spacing,
                    train_transforms=True,
                    augm_params=augm_params
                )

            valid_transforms = build_transforms_dynunet(
                keys=['image', 'mask'],
                patch_size=patch_size,
                target_spacing=median_spacing,
                train_transforms=False, 
                augm_params=augm_params
            )

            batch_size = auto_batch_size
        
        else:
            spatial_dims = net_params.get('spatial_dims', 2)

            train_transforms = build_transform_list(['filepath'],
                                                    median_spacing,
                                                    True,
                                                    augm_params,
                                                    spatial_dims)

            valid_transforms = build_transform_list(['filepath'],
                                                    median_spacing,
                                                    False,
                                                    augm_params,
                                                    spatial_dims)

        self.model_params['out_channels'] = n_classes
        self.train_params['batch_size'] = batch_size
        
        return model, train_transforms, valid_transforms


    def freeze_layers(self, model, degree: float) -> None:
        '''
        Freeze a percentage of the model layers
        
        :param model: model to freeze layers
        :param degree: percentage of layers to freeze
        '''

        named_params = list(model.named_parameters())
        num_params = len(named_params)
        num_to_freeze = int(num_params * degree)

        self.sig_status.emit(f"Fine-tuning: freezing {num_to_freeze}/{num_params} \
            parameter blocks ({degree*100:.0f}%)")
        
        for i, (name, param) in enumerate(named_params):
            if i < num_to_freeze:
                if "norm" in name.lower() or 'bn' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True


    # implementation of run method that will be run in separate Thread
    def run(self):
        '''
            Thread entry point.
            Contains the sequential training logic.
        '''
        
        try:
            from .dafne_dataset import DafneDataset

            self.sig_status.emit(f"Training initialization on: {self.device} device")
            self.sig_status.emit(f"Dataset loading ({len(self.file_list)} files...) in {self.file_list}")
            
            # split dataset into train and validation
            train_list, valid_list = train_test_split(self.file_list, test_size=0.2, random_state=42)
            
            n_classes = count_label_mask(data_list=self.file_list)
            self.model_params['n_classes'] = n_classes
            self.model_params['out_channels'] = n_classes

            augm_params = self.train_params.get('augmentation', {}) #read augmentation parameters
            
            #fine-tuning mode
            if self.pretrained_model_path:
                self.sig_status.emit("Fine-tuning mode")

                loaded_obj = DynamicTorchModel.Load(self.pretrained_model_path)
                model = loaded_obj.model

                # Get percent_to_freeze from train_params if available
                percent_to_freeze = self.train_params.get('percent_to_freeze', None)
                lora_config = self.train_params.get('lora_config', None)

                model, train_transforms, valid_transforms = \
                        self._setup_fine_tuning(loaded_obj, 
                                                percent_to_freeze,
                                                lora_config,
                                                n_classes)               

            else: 
                model, train_transforms, valid_transforms = \
                    self._setup_training()               


            train_dataset = DafneDataset(data_files=train_list,
                                        augm_params=augm_params,
                                        train_transform=True,
                                        spatial_dims=self.model_params.get('spatial_dims', 2),
                                        external_transforms=train_transforms
                                        )
            valid_dataset = DafneDataset(data_files=valid_list,
                                        augm_params={},
                                        train_transform=False,
                                        spatial_dims=self.model_params.get('spatial_dims', 2),
                                        external_transforms=valid_transforms
                                        )
            
            # batch size must be choose by user before train
            train_dataloader = DataLoader(train_dataset, 
                                          num_workers=8, 
                                          batch_size=self.model_params.get('batch_size', 1), 
                                          shuffle=True,
                                          collate_fn=pad_list_data_collate)
            
            valid_dataloader = DataLoader(valid_dataset, 
                                          num_workers=8, 
                                          batch_size=1, 
                                          shuffle=False,
                                          collate_fn=pad_list_data_collate)
            
            # Filter parameters to only optimize those with requires_grad=True
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                        lr=self.train_params.get('learning_rate', 0.001))
            
            # define loss criterion
            criterion = DiceCELoss(include_background=False, 
                                 softmax=True,
                                 to_onehot_y=True)

            pytorch_training_loop(
                model=model,
                optimizer=optimizer,
                train_dataloader = train_dataloader,
                valid_dataloader = valid_dataloader,
                criterion=criterion,
                device=self.device,
                epochs=self.train_params.get('epochs', 100),
                #callback injection
                save_path=self.save_path,
                on_epoch_end=self._callback_epoch_end,
                check_stop=self._callback_check_stop,
                on_log=self._callback_log,
                early_stopping=self.early_stopping,
                n_classes=n_classes,
                spatial_dims=self.model_params['spatial_dims'],
                val_roi_size=self.model_params['patch_size'],
                model_name=self.model_params['model_name']
            )

            save_dir = os.path.dirname(self.save_path)
            '''
            json_path = os.path.join(save_dir, f'{model_name}_params.json')
            json_memory_buffer_path = os.path.join(save_dir, f'{model_name}_memory_buffer.json')
            '''

            # to be added: labels' name, norm_params, version
            save_params = {
                'model_name': self.model_params.get('model_name', 'unet'),
                'train_list': train_list,
                'valid_list': valid_list,
                'spatial_dims': self.model_params.get('spatial_dims', 2),
                'n_levels': self.model_params.get('n_levels', 5),
                'kernel_size': self.model_params.get('kernel_size', 3),
                'out_channels': self.model_params.get('out_channels', 1),
                'in_channels': self.model_params.get('in_channels', 1),
                'median_spacing': self.model_params.get('median_spacing', None),
                'use_dynamic': self.model_params.get('use_dynamic', False),
                'kernels': self.model_params.get('kernels', None),
                'strides': self.model_params.get('strides', None),
                'batch_size': self.train_params.get('batch_size', 1),
                'learning_rate': self.train_params.get('learning_rate', 0.001),
                'patch_size': self.model_params.get('patch_size', None),
                'data_shape': self.model_params.get('data_shape', None)
            }

            # take random images from train and validation list to save memory buffer
            # take random 20% images from original train_list and validation_list
            memory_buffer = {
                'train_path_list': [os.path.abspath(p) for p in rd.sample(train_list, len(train_list)//20)],
                'valid_path_list': [os.path.abspath(p) for p in rd.sample(valid_list, len(valid_list)//20)]
            }

            '''
            try:
                with open(json_path, "w") as json_data: 
                    json.dump(save_params, json_data, indent=4)
            except Exception as e:
                print(f"Error saving params: {e}")
            
            try:
                with open(json_memory_buffer_path, "w") as json_data:
                    json.dump(memory_buffer, json_data, indent=4)
            except Exception as e:
                print(f"Error saving memory buffer: {e}")
            '''

            #save model, weights and metadata in .dafne format
            weights_path = os.path.join(save_dir, f'{self.model_params.get("model_name", "unet")}_best_model.pth')
            best_weights = torch.load(weights_path, map_location='cpu')
            output_path = os.path.join(save_dir, f'{self.model_params.get("model_name", "unet")}_final_model.dafne')
            self.sig_status.emit("Packaging the model into .dafne format...")
            with open(output_path, 'wb') as f:
                create_dynamic_model(weights=best_weights, net_metadata=save_params, train_metadata=memory_buffer).dump(f)
            
            if not self.is_running:
                self.sig_stopped.emit()
            else:
                self.sig_finished.emit()
            
        except Exception as e:
            traceback.print_exc()
            self.sig_error.emit(str(e))

    def stop(self):
        self.is_running = False            