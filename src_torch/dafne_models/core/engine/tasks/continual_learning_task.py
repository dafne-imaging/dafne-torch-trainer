from .supervised_task import SupervisedModelTask

import torch
import os

def compute_ewc_loss(model, fisher, params_snapshot, lambda_reg):
    ewc_loss = 0.0
    for name, param in model.named_parameters():
        if name in fisher and name in params_snapshot:
             ewc_loss += (fisher[name] * (param - params_snapshot[name]) ** 2).sum()
    return lambda_reg * ewc_loss

class ContinualLearningTask(SupervisedModelTask):

    def __init__(self, model,
                 criterion,
                 optimizer, 
                 device,
                 spatial_dims: int,
                 val_roi_size: tuple,
                 mixed_precision: bool = False,
                 save_path: str = None,
                 lambda_reg: float = 1.0,
                 ):

        super().__init__(model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         device=device,
                         spatial_dims=spatial_dims, 
                         val_roi_size=val_roi_size,
                         mixed_precision=mixed_precision)
        
        self.model = model
        self.lambda_reg = lambda_reg
        self.ewc_params = torch.load(os.path.join(os.path.dirname(save_path), '_ewc.pt'), weights_only=True)

    def train_step(self, engine, batch):
        self.model.train() #pre-trained model
        inputs = batch['image'].to(self.device)
        targets = batch['mask'].long().to(self.device)
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) #ewc loss
                loss = loss + compute_ewc_loss(self.model, 
                                               self.ewc_params.get('fisher'), 
                                               self.ewc_params.get('params_snapshot'), lambda_reg=self.lambda_reg)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #gradient clipping
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {'loss': loss.item()}