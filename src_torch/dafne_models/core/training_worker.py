import sys
import os
import json
import traceback
import random as rd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from monai.data import DataLoader, decollate_batch
from monai.data.utils import pad_list_data_collate, list_data_collate
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, AsDiscrete

from PyQt5.QtCore import QThread, pyqtSignal

from sklearn.model_selection import train_test_split

from .utils import count_label_mask, get_median_spacing
from ..utils.data_fingerprint import DatasetFingerprint
from ..utils.optimizer import get_optimal_hyperparameters
from ..models import dafne_networks
from .transforms_builder import (build_transform_list, 
                                 build_transforms_dynunet)

PATIENT = 20

# definition of classic training loop
def pytorch_training_loop(model, 
                          train_dataloader,
                          valid_dataloader,
                          optimizer, 
                          criterion,
                          device,
                          epochs,
                          spatial_dims:int=2,
                          n_classes:int=2,
                          early_stopping:bool=False,
                          save_path:str=None,
                          on_epoch_end=None,
                          check_stop=None,
                          on_log=None,
                          val_roi_size=None
                          ):
    
    if on_log: on_log(f"Engine Starting on device {device}. {epochs} epochs")
    
    dice_metric = DiceMetric(include_background=True, reduction='mean')
    best_val_dice_score = -float("inf")
    counter = 0

    for epoch in range(epochs):
        
        # check training stop  by user
        if check_stop is not None and check_stop():
            if on_log: on_log(f"Training stopped by user")
            break
        
        # classic pytorch training loop defined
        model.train()
        epoch_loss = 0.0
        
        for batch in train_dataloader:
            if check_stop is not None and check_stop():
                break
            inputs = batch['image'].to(device)
            targets = batch['mask'].long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        if check_stop is not None and check_stop():
            break
        
        avg_loss = epoch_loss / len(train_dataloader)

        val_loss = 0.0
        model.eval()

        post_pred = Compose([
            AsDiscrete(argmax=True, to_onehot=n_classes)
        ])
        post_label = Compose([
            AsDiscrete(to_onehot=n_classes)
        ])

        with torch.no_grad():
            for batch in valid_dataloader:
                if check_stop is not None and check_stop():
                    break
                val_image = batch['image'].to(device)
                val_mask = batch['mask'].long().to(device)
                if spatial_dims == 3: 
                    val_output = valid_on_batch_3d(val_image, model, val_roi_size) # return original tensor shape
                else:
                    val_output = model(val_image)
                val_output_decollate = decollate_batch(val_output)
                val_label_decollate = decollate_batch(val_mask)
                val_preds = [post_pred(i) for i in val_output_decollate]
                val_masks = [post_label(i) for i in val_label_decollate]
                
                dice_metric(y_pred=val_preds, y=val_masks)
                val_loss += criterion(val_output, val_mask).item()
            
            if check_stop is not None and check_stop():
                break

            dice_score = dice_metric.aggregate().item()
            avg_val_loss = val_loss / len(valid_dataloader)

            dice_metric.reset()

            if dice_score > best_val_dice_score:
                best_val_dice_score = dice_score
                counter = 0
            
                if save_path:
                    try: 
                        save_dir = os.path.dirname(save_path)
                        best_model_path = os.path.join(save_dir, '_best_model.pth')
                        torch.save(model.state_dict(), best_model_path)
                        
                        if on_log: 
                            on_log(f'New best Dice score {best_val_dice_score:.4f}. Model saved!')
                            
                    except Exception as e: 
                        print(f'Error during model saving: {e}')
            
            elif early_stopping:
                counter += 1       
                if counter >= PATIENT:
                    on_log(f'Training interrupted because of early stopping. Model saved in {save_path}') 
                    break
                
        #take random valid image of batch from random valid dataloader
        dataset = valid_dataloader.dataset
        idx = rd.randrange(len(dataset))
        sample = dataset[idx]
        val_image = sample['image'].to(device)
        val_mask = sample['mask'].to(device)

        model.eval()
        with torch.no_grad():
            if spatial_dims == 3: 
                val_pred_out = valid_on_batch_3d(val_image.unsqueeze(0), model, val_roi_size)
            else: 
                val_pred_out = model(val_image.unsqueeze(0)) # add batch dim
            val_pred = torch.argmax((val_pred_out), dim=1).float()

            dims = val_image.shape
            if len(dims)==4: # 3D volume [B, H, W, D] - no C because of argmax
                img_np = val_image[0, dims[1]//2, :, :].cpu().numpy()
                pred_np = val_pred[0, dims[1]//2, :, :].cpu().numpy()
            elif len(dims)==3: # 2D images [B, H, W] - no C because of argmax
                img_np = val_image[0].cpu().numpy()
                pred_np = val_pred[0].cpu().numpy()

            # send to GUI for each epoch, the current epoch, avg_loss and rand image
            # and his model predicted mask
            if on_epoch_end:    
                on_epoch_end(epoch, avg_loss, img_np, pred_np, avg_val_loss)
       
    if on_log: on_log(f'Trainging engine finished. Best Dice {best_val_dice_score:.4f}')


def valid_on_batch_3d(image_batch, model, val_roi_size):
    '''
    Methods for validation on 3D batches with sliding inference
    
    :param image_batch: current image batch
    :param model: model for patch prediction
    '''
    return sliding_window_inference(
        inputs=image_batch,
        roi_size=val_roi_size,
        sw_batch_size=4,
        overlap=0.25,
        predictor=model
    )


class TrainingWorker(QThread):

    '''
        Worker class that runs the PyTorch training loop in a separate thread.
        Ensures the GUI remains responsive during intensive computation.
    '''
    
    # Send data to cpu (float, numpy, numpy)
    sig_update_plot = pyqtSignal(float, object, object, float)

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
                 mask_list:list=None,
                 save_path:str=None,
                 early_stopping:bool=False,
                 dyn_model_params:dict=None,
                 ):
        super().__init__()
        
        # inzialize worker parameters
        self.file_list = file_list
        self.model_params = model_params
        self.dyn_unet_params = dyn_model_params
        self.train_params = train_params
        self.save_path = save_path

        self.is_running = True
        self.early_stopping = early_stopping

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _callback_epoch_end(self, epoch, loss, img, mask, val_loss):
        
        '''
        Function called at the end of each epoch by the Engine
        '''

        self.sig_update_plot.emit(loss, img, mask, val_loss)
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
            
            #n_classes = count_label_mask(train_list, spatial_dims=self.model_params.get('spatial_dims', 2))
            # define model that has be trained
            # this is an example of unet model
            spatial_dims = self.model_params.get('spatial_dims', 2)
            use_dynamic = self.model_params.get('use_dynamic', False)
            augm_params = self.train_params.get('augmentation', {})
            median_spacing = get_median_spacing(self.file_list, spatial_dims)

            spatial_dims = self.model_params.get('spatial_dims', 3)
            n_classes = count_label_mask(data_list=self.file_list)

            if use_dynamic and spatial_dims == 3:
                self.sig_status.emit("Dynamic Mode: Analyzing Dataset Fingerprint...")

                fingerprint = DatasetFingerprint(self.file_list, spatial_dims=3)
                median_spacing = fingerprint.data_spacing
                median_shape = fingerprint.data_shape

                patch_size, auto_batch_size = get_optimal_hyperparameters(
                    median_shape, spatial_dims=3
                )
                final_patch_size = patch_size
                
                self.sig_status.emit(f"Auto-Config: Patch={patch_size}, Batch={auto_batch_size}")
                kernels, strides = fingerprint.get_kernel_and_strides(patch_size)

                model = dafne_networks.DafneDynUnet(spatial_dims = 3,
                                                    in_channels=self.model_params.get('in_channels', 1),
                                                    out_channels=n_classes,
                                                    kernel_size=kernels,
                                                    strides=strides,
                                                    norm_name=("INSTANCE", {"affine": True}),
                                                    deep_supervision=False).to(self.device)
                
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
                self.sig_status.emit("Classic Mode: Using Standard U-Net")
                
                fingerprint = DatasetFingerprint(self.file_list, spatial_dims=3)
                median_spacing = fingerprint.data_spacing

                model = dafne_networks.DafneUnetModel(spatial_dims=spatial_dims,
                                                 n_levels=self.model_params.get('n_levels', 5),
                                                 kernel_size=self.model_params.get('kernel_size', 3),
                                                 out_channels=n_classes,
                                                 in_channels=self.model_params.get('in_channels', 1)).to(self.device)
                
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
            
            train_dataset = DafneDataset(data_files=train_list,
                                        augm_params=augm_params,
                                        train_transform=True,
                                        spatial_dims=spatial_dims,
                                        external_transforms=train_transforms
                                        )
            valid_dataset = DafneDataset(data_files=valid_list,
                                        augm_params={},
                                        train_transform=False,
                                        spatial_dims=spatial_dims,
                                        external_transforms=valid_transforms
                                        )
            
            # batch size must be choose by user before train
            batch_size = self.train_params.get('batch_size', 2)
            train_dataloader = DataLoader(train_dataset, 
                                          num_workers=8, 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          collate_fn=pad_list_data_collate)
            
            valid_dataloader = DataLoader(valid_dataset, 
                                          num_workers=8, 
                                          batch_size=1, 
                                          shuffle=False,
                                          collate_fn=pad_list_data_collate)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params.get('learning_rate', 0.001))
            
            # define loss criterion
            criterion = DiceLoss(include_background=True, 
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
                spatial_dims=spatial_dims,
                val_roi_size=final_patch_size if use_dynamic else (16, 96, 96)
            )

            save_dir = os.path.dirname(self.save_path)
            json_path = os.path.join(save_dir, '_params.json')

            save_params = {
                'spatial_dims': self.model_params.get('spatial_dims', 2),
                'n_levels': self.model_params.get('n_levels', 5),
                'kernel_size': self.model_params.get('kernel_size', 3),
                'out_channels': n_classes,
                'in_channels': self.model_params.get('in_channels', 1),
                'median_spacing': median_spacing.tolist() if isinstance(median_spacing, np.ndarray) else median_spacing,
                'use_dynamic': use_dynamic,
                'kernel': kernels if use_dynamic else None,
                'strides': strides if use_dynamic else None
            }

            try:
                with open(json_path, "w") as json_data: 
                    json.dump(save_params, json_data, indent=4)
            except Exception as e:
                print(f"Error saving params: {e}")
            
            if not self.is_running:
                self.sig_stopped.emit()
            else:
                self.sig_finished.emit()
            
        except Exception as e:
            traceback.print_exc()
            self.sig_error.emit(str(e))

    def stop(self):
        self.is_running = False            