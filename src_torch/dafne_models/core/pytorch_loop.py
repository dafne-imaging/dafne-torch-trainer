import torch
import os
import random as rd
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import  (DiceMetric, 
                            HausdorffDistanceMetric, 
                            SurfaceDistanceMetric, 
                            ConfusionMatrixMetric,
                            compute_confusion_matrix_metric)
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, AsDiscrete

PATIENT = 20

MONAI_REGISTRY = {
    'hausdorff_95': HausdorffDistanceMetric,
    'surface_distance': SurfaceDistanceMetric,
    'precision': ConfusionMatrixMetric,
    'sensitivity': ConfusionMatrixMetric,
    'threat_score': ConfusionMatrixMetric,
    'specificity': ConfusionMatrixMetric,
    'accuracy': ConfusionMatrixMetric,
    'f1_score': ConfusionMatrixMetric,
}

# definition of classic training loop
# with FP16 precision and autocast
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
                          scheduler=None,
                          mixed_precision:bool=False,
                          save_path:str=None,
                          model_name:str=None,
                          on_epoch_end=None,
                          check_stop=None,
                          on_log=None,
                          val_roi_size=None,
                          labels_name:list[str]=None,
                          inference_metrics=None
                          ):
    
    if on_log: on_log(f"Engine Starting on device {device}. {epochs} epochs")
    
    dice_metric = DiceMetric(include_background=False, reduction='mean_batch')
    best_val_dice_score = -float("inf")
    counter = 0

    history_metrics = []
    active_metrics = {}
    metrics_to_log = {'epoch': 0, 
                        'train_loss': 0.0, 
                        'val_loss': 0.0,
                        'dice_avg': 0.0}
    for key, value in inference_metrics.items():
        if key.startswith('compute_') and value:
            metric = key.replace('compute_', '')
            if metric in MONAI_REGISTRY:
                if isinstance(MONAI_REGISTRY[metric], ConfusionMatrixMetric):
                    metric_monai_name = metric.replace('_', ' ')
                    active_metrics[metric] = MONAI_REGISTRY[metric](
                        include_background=inference_metrics['include_background'], 
                        reduction=inference_metrics['reduction'],
                        metric_name=metric_monai_name
                    )
                else:
                    active_metrics[metric] = MONAI_REGISTRY[metric](
                        include_background=inference_metrics['include_background'], 
                        reduction=inference_metrics['reduction']
                    )
                metrics_to_log[f'avg_{metric}'] = 0.0
                if labels_name:
                    for label in labels_name:
                        metrics_to_log[f'{metric}_{label}'] = 0.0
            else:
                if on_log: on_log(f"Metric {metric} not found in MONAI_REGISTRY")
    if labels_name:
        for label in labels_name:
            metrics_to_log[f'dice_{label}'] = 0.0
    if save_path:
        log_dir = os.path.join(os.path.dirname(save_path), "logs")
        tb_writer = SummaryWriter(log_dir=log_dir)
    else:
        tb_writer = None

    scaler = torch.amp.GradScaler(enabled=mixed_precision)

    for epoch in range(epochs):
        
        # check training stop  by user
        if check_stop is not None and check_stop():
            if on_log: on_log(f"Training stopped by user")
            break
        
        epoch_loss = 0.0
        avg_loss = 0.0
        avg_val_loss = 0.0
        per_mask_dice_score = {}
        metrics_to_log = metrics_to_log.copy()
        
        metrics_to_log['epoch'] = epoch + 1
        metrics_to_log['train_loss'] = 0.0
        metrics_to_log['val_loss'] = 0.0
        metrics_to_log['dice_avg'] = 0.0
        
        model.train()
        for batch in train_dataloader:
            if check_stop is not None and check_stop():
                break
            inputs = batch['image'].to(device)
            targets = batch['mask'].long().to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
        
        if scheduler is not None:
            #current_lr = scheduler.get_last_lr()
            scheduler.step()

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

        if torch.cuda.is_available():
            try:
                del_from_gpu(inputs, targets)
            except NameError:
                pass
        
        per_mask_dice_score = {}
        with torch.no_grad():
            for batch in valid_dataloader:
                if check_stop is not None and check_stop():
                    break
                val_image = batch['image'].to(device)
                val_mask = batch['mask'].long().to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
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

                for metric in active_metrics.values():
                    metric(y_pred=val_preds, y=val_masks)
            
            if check_stop is not None and check_stop():
                break
            
            # compute dice score metrics
            dice_score = dice_metric.aggregate() # mean of all batches for each mask
            dice_score_avg = dice_score.mean().item() # mean of all masks
            avg_val_loss = val_loss / len(valid_dataloader)
            per_mask_dice_score = {name: dice_score[i].item() for i, name in enumerate(labels_name)}
            dice_metric.reset()

            for metric_name, metric in active_metrics.items():
                metric_score = metric.aggregate()

                # In case metric.aggregate() returns a list across multiple batches, concatenate them
                if isinstance(metric_score, list):
                    metric_score = torch.cat(metric_score, dim=0)

                metric_score_avg = metric_score.mean().item()
                per_mask_metric_score = {name: metric_score[i].item() for i, name in enumerate(labels_name)}
                metrics_to_log[f'avg_{metric_name}'] = metric_score_avg
                for label in labels_name:
                    metrics_to_log[f'{metric_name}_{label}'] = per_mask_metric_score[label]
                metric.reset()

            metrics_to_log['train_loss'] = avg_loss
            metrics_to_log['val_loss'] = avg_val_loss
            metrics_to_log['dice_avg'] = dice_score_avg

            if labels_name:
                for label in labels_name:
                    metrics_to_log[f'dice_{label}'] = per_mask_dice_score[label]

            if dice_score_avg > best_val_dice_score:
                best_val_dice_score = dice_score_avg
                counter = 0
            
                if save_path:
                    try: 
                        save_dir = os.path.dirname(save_path)
                        filename = f"{model_name}_best_model.pth" if model_name else "_best_model.pth"
                        best_model_path = os.path.join(save_dir, filename)
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
            
            if torch.cuda.is_available():
                del_from_gpu(val_output, val_mask, val_preds, val_masks)
                
        #take random valid image of batch from random valid dataloader
        dataset = valid_dataloader.dataset
        idx = rd.randrange(len(dataset))
        sample = dataset[idx]
        val_image = sample['image'].to(device)
        val_mask = sample['mask'].to(device)

        current_spacing = sample['image_meta_dict']['pixdim']
        if isinstance(current_spacing, torch.Tensor):
            current_spacing = current_spacing.cpu().numpy()
        img_spacing = current_spacing[1:]

        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
                if spatial_dims == 3: 
                    val_pred_out = valid_on_batch_3d(val_image.unsqueeze(0), model, val_roi_size)
                else: 
                    val_pred_out = model(val_image.unsqueeze(0)) # add batch dim
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

            # send to GUI for each epoch, the current epoch, avg_loss and rand image
            # and his model predicted mask
            if on_epoch_end:    
                on_epoch_end(epoch, 
                            avg_loss, 
                            img_np, pred_np, 
                            avg_val_loss, 
                            img_spacing, 
                            best_val_dice_score,
                            per_mask_dice_score
                            )
        
        if tb_writer:
            for key, val in metrics_to_log.items():
                if key == 'epoch' or val is None: 
                    continue
                # Organization tags for cleaner TensorBoard UI
                tag = key.replace('train_', 'Loss/').replace('val_', 'Loss/').replace('dice_', 'Dice/')
                tb_writer.add_scalar(tag, val, epoch)
            
            # Optional: log the preview image to TensorBoard as well
            if 'img_np' in locals() and 'pred_np' in locals():
                # Normalized images for display
                tb_writer.add_image('Preview/Image', img_np, epoch, dataformats='HW')
                tb_writer.add_image('Preview/Prediction', pred_np, epoch, dataformats='HW')

    if on_log: on_log(f'Trainging engine finished. Best Dice {best_val_dice_score:.4f}')

    # delete tensors and model from gpu memory
    if torch.cuda.is_available():
        model.to('cpu')
        try:
            del_from_gpu(model, val_output, val_mask, val_preds, val_masks)
        except NameError:
            # handle case where some variables might not be defined if training stopped early
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    if tb_writer:
        tb_writer.close()

    return float(best_val_dice_score)


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

def del_from_gpu(*model_params):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
