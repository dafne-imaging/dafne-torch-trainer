from .base_task import BaseTask
import os

import torch
import random as rd
from monai.inferers import sliding_window_inference

class SupervisedModelTask(BaseTask):
    def __init__(self, 
                 model,
                 criterion,
                 optimizer, 
                 device,
                 spatial_dims: int,
                 val_roi_size: tuple,
                 mixed_precision: bool = False
                 ):
        """
        Initialize the task.
        """
        super().__init__()
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.spatial_dims = spatial_dims
        self.val_roi_size = val_roi_size

        self.scaler = torch.amp.GradScaler(enabled=mixed_precision)
        self.mixed_precision = mixed_precision
    
    def train_step(self, engine, batch):
        self.model.train()
        inputs = batch['image'].to(self.device)
        targets = batch['mask'].long().to(self.device)
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #gradient clipping
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {'loss': loss.item()}

    def validation_step(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            val_image = batch['image'].to(self.device)
            val_mask = batch['mask'].long().to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
                if self.spatial_dims == 3: 
                    val_output = self.valid_on_batch_3d(val_image) # return original tensor shape
                else:
                    val_output = self.model(val_image)
                val_loss = self.criterion(val_output, val_mask)
        
        patient_name = os.path.basename(batch['filepath'][0])
        slice_idx = batch['index'].item() if 'index' in batch else '3D'
        engine.state.metadata.append({
            'patient': patient_name,
            'slice': slice_idx
        })
        
        return {
            'loss': val_loss.item(),
            'y_pred': val_output,
            'y': val_mask
        }
    
    def compute_predictions(self, engine, random_batch):
        val_image = random_batch['image'].to(self.device)
        val_mask = random_batch['mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
                if self.spatial_dims == 3: 
                    val_pred_out = self.valid_on_batch_3d(val_image.unsqueeze(0))
                else: 
                    val_pred_out = self.model(val_image.unsqueeze(0)) # add batch dim
            val_pred = torch.argmax((val_pred_out), dim=1).float()

            dims = val_image.shape
            if len(dims)==4: # 3D volume [C, D, H, W]
                depth = dims[1]
                slice_idx = rd.randint(int(depth * 0.1), int(depth * 0.9))
                img_np = val_image[0, slice_idx, :, :].cpu().numpy()
                pred_np = val_pred[0, slice_idx, :, :].cpu().numpy()
            elif len(dims)==3: # 2D images [B, H, W] - no C because of argmax
                img_np = val_image[0].cpu().numpy()
                pred_np = val_pred[0].cpu().numpy()
        
        return {
            'image': img_np,
            'mask': pred_np,
            'y': val_mask.cpu().numpy()
        }
    
    def valid_on_batch_3d(self, image_batch):
        '''
        Methods for validation on 3D batches with sliding inference
        
        Args: 
            image_batch: image batch to evaluate
        Return: 
            Prediction output
        '''
        return sliding_window_inference(
            inputs=image_batch,
            roi_size=self.val_roi_size,
            sw_batch_size=4,
            overlap=0.25,
            predictor=self.model
        )