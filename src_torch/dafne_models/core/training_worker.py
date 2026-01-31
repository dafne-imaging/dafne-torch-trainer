# ------- Load dependecies ------------------ # 
import sys
import traceback
import random as rd

import torch
import torch.nn as nn
import torch.optim as optim

from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from PyQt5.QtCore import QThread, pyqtSignal

from sklearn.model_selection import train_test_split

from ..models import dafne_network

# definition of classic training loop
def pytorch_training_loop(model, 
                          train_dataloader,
                          valid_dataloader,
                          optimizer, 
                          criterion,
                          device,
                          epochs,
                          on_epoch_end=None,
                          check_stop=None,
                          on_log=None):
    
    if on_log: on_log(f"Engine Starting on device {device}. {epochs} epochs")
    
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    best_val_dice_score = 0.0

    for epoch in range(epochs):
        
        # check training stop  by user
        if check_stop is not None and check_stop():
            if on_log: on_log(f"Training stopped by user")
            break
        
        # classic pytorch training loop defined
        model.train()
        epoch_loss = 0.0
        
        for batch in train_dataloader:
            inputs = batch['image'].to(device)
            targets = batch['mask'].long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)

        with torch.no_grad():
            for batch in valid_dataloader:
                val_image = batch['image']
                val_mask = batch['mask']
                if on_epoch_end is not None:
                    model.eval()
                    val_output = model(val_image)
                    val_pred = torch.argmax(val_output, dim=1, keepdim=True)
                    dice_metric(y_pred=val_pred, y=val_mask)
            dice_score = dice_metric.aggregate().item()
            dice_metric.reset()

            if dice_score > best_val_dice_score:
                best_val_dice_score = dice_score
            
            #take first valid image for first batch to display during training
            sample_batch = next(iter(valid_dataloader))
            val_image = sample_batch['image'][0].to(device)
            val_mask = sample_batch['mask'][0].to(device)
            val_pred_out = model(val_image.unsqueeze(0))
            val_pred = torch.argmax((val_pred_out), dim=1).float()

            dims = val_image.shape
            if len(dims)==5:
                img_np = val_image[0, :, :, dims[3]//2].cpu().numpy()
                pred_np = val_pred[0, :, :, dims[3]//2].cpu().numpy()
            elif len(dims)==4:
                img_np = val_image[0, :, :].cpu().numpy()
                mask_np = val_mask[0, 0, :].cpu().numpy()
                pred_np = val_pred[0, :, :].cpu().numpy()

                # send to GUI for each epoch, the current epoch, avg_loss and rand image
                # and his model predicted mask
            if on_epoch_end:    
                on_epoch_end(epoch, avg_loss, img_np, pred_np)

        '''if on_epoch_end is not None:
            # preview during trainging results
            model.eval()
            sample_in = inputs[0].unsqueeze(0)
            sample_out = model(sample_in)
            pred_mask = torch.argmax(sample_out, dim=1)
            with torch.no_grad():
                dims = inputs.shape
                if len(dims)==5: #[B, C, H, W, D]
                    img_np = inputs[0, 0, :, :, dims[4]//2].cpu().numpy()
                    mask_np = pred_mask[0, 0, :, :, dims[4]//2].cpu().numpy()
                elif len(dims)==4:
                    img_np = inputs[0].cpu().numpy()[0]
                    mask_np = pred_mask.cpu().numpy()[0] 
                
                on_epoch_end(epoch, avg_loss, img_np, mask_np)'''
        
    if on_log: on_log(f'Trainging engine finished. Best Dice {best_val_dice_score:.4f}')


class TrainingWorker(QThread):

    '''
        Worker class that runs the PyTorch training loop in a separate thread.
        Ensures the GUI remains responsive during intensive computation.
    '''
    
    # Send data to cpu (float, numpy, numpy)
    sig_update_plot = pyqtSignal(float, object, object)

    # Send status information for user console
    sig_status = pyqtSignal(str)

    # Send error information to user console
    sig_error = pyqtSignal(str)

    # progress values to sent to GUI
    sig_progress = pyqtSignal(int)

    # Send message when training ended
    sig_finished = pyqtSignal()

    def __init__(self,
                 file_list:list,
                 model_params:dict,
                 train_params:dict, 
                 mask_list:list=None,
                 save_path:str=None):
        super().__init__()
        
        # inzialize worker parameters
        self.file_list = file_list
        self.mask_list = mask_list
        self.model_params = model_params
        self.train_params = train_params
        self.save_path = save_path

        self.is_running = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _callback_epoch_end(self, epoch, loss, img, mask):
        
        '''
        Function called at the end of each epoch by the Engine
        '''

        self.sig_update_plot.emit(loss, img, mask)
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

    # implementation of run method that will be run in separate Thread
    def run(self):
        '''
            Thread entry point.
            Contains the sequential training logic.
        '''
        
        try:
            from .dafne_dataset import DafneCacheDataset

            self.sig_status.emit(f"Training initialization on: {self.device} device")
            self.sig_status.emit(f"Dataset loading ({len(self.file_list)} files...)")
            
            # define model that has be trained
            # this is an example of unet model
            model = dafne_network.DafneUnetModel(spatial_dims=self.model_params.get('spatial_dims', 2),
                                                 n_levels=self.model_params.get('n_levels', 5),
                                                 kernel_size=self.model_params.get('kernel_size', 3),
                                                 out_channels=self.model_params.get('n_classes', 2),
                                                 in_channels=self.model_params.get('in_channels', 1)).to(self.device)
            # split dataset into train and validation
            if self.mask_list is None:
                train_list, valid_list = train_test_split(self.file_list, test_size=0.2, random_state=42)
                train_mask, valid_mask = None, None
            else: 
                train_list, valid_list, train_mask, valid_mask = train_test_split(self.file_list, 
                                                                              test_size=0.2, random_state=42)
            
            train_dataset = DafneCacheDataset(image_files=train_list,
                                        mask_files=train_mask,
                                        cache_rate=1.0)
            valid_dataset = DafneCacheDataset(image_files=valid_list,
                                        mask_files=valid_mask,
                                        cache_rate=1.0)
            dataset = DafneCacheDataset(image_files=self.file_list,
                                        mask_files=self.mask_list,
                                        cache_rate=1.0)
            
            # batch size must be choose by user before train
            #dataloader = DataLoader(dataset, num_workers=0, batch_size=self.train_params.get('batch_size', 2))
            train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=self.train_params.get('batch_size', 2), shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, num_workers=0, batch_size=self.train_params.get('batch_size', 2), shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params.get('learning_rate', 0.001))
            
            # define loss criterion
            # define dice loss as criterion
            #criterion = nn.CrossEntropyLoss()
            criterion = DiceLoss(include_background=False, 
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
                on_epoch_end=self._callback_epoch_end,
                check_stop=self._callback_check_stop,
                on_log=self._callback_log
            )

            if self.save_path: 
                self.sig_status.emit(f"Saving model to {self.save_path}...")
                torch.save(model.state_dict(), self.save_path)
                print(f'Model weights saved in {self.save_path}')
            self.sig_finished.emit()
            
        except Exception as e:
            traceback.print_exc()
            self.sig_error.emit(str(e))

    def stop(self):
        self.is_running = False            