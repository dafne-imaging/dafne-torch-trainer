import logging
import torch
import os
import csv

logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter

from .base_callback import BaseCallback
from typing import Dict, Any

from monai.metrics import ConfusionMatrixMetric
import random as rd
from monai.data import decollate_batch

class CSVLoggingCallback(BaseCallback):
    '''
    CSV logging callback.
    Saves per-patient scores for all active metrics (auto-detected from engine.state.metrics buffer_ keys).
    '''
    def __init__(self, save_path: str, labels_name: list = None):
        super().__init__()
        self.csv_path = save_path.replace('.model', '.csv') if save_path else None
        self.labels_name = labels_name or []
        self._best = -float('inf')

    def on_epoch_completed(self, engine):
        if not self.csv_path or not engine.state.metadata:
            return

        if engine.state.best_metric > self._best:
            self._best = engine.state.best_metric

            # auto-detect all metrics that have a per-patient buffer
            active_metrics = [
                key[len('buffer_'):]
                for key in engine.state.metrics
                if key.startswith('buffer_')
            ]

            header = ['Patient', 'Slice']
            for m_name in active_metrics:
                for label in self.labels_name:
                    header.append(f"{m_name}_{label}")

            buffers = {
                m: engine.state.metrics.get(f"buffer_{m}")
                for m in active_metrics
            }

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, meta in enumerate(engine.state.metadata):
                    row = [meta['patient'], meta['slice']]
                    for m_name in active_metrics:
                        buf = buffers[m_name]
                        if buf is not None and i < len(buf):
                            row.extend([f"{score.item():.4f}" for score in buf[i]])
                        else:
                            row.extend([""] * len(self.labels_name))
                    writer.writerow(row)

            logger.info("CSV report saved in: %s", self.csv_path)


class TensorboardCallback(BaseCallback):
    def __init__(self, save_path: str):
        '''
        Args:
            save_path: Model save path, used to derive the log directory structure.
                       If None, TensorBoard logging is disabled.
        '''
        super().__init__()
        if save_path:
            model_filename = os.path.basename(save_path).split('.')[0]
            log_dir = os.path.join(os.path.dirname(save_path), "logs", model_filename)
            self.writer_train = SummaryWriter(log_dir=os.path.join(log_dir, "train"))
            self.writer_val   = SummaryWriter(log_dir=os.path.join(log_dir, "val"))
        else:
            self.writer_train = None
            self.writer_val = None

    def on_epoch_completed(self, engine):
        if not self.writer_train or not self.writer_val:
            return
        epoch = engine.state.epoch

        for name, value in engine.state.metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if name == 'avg_loss':
                self.writer_train.add_scalar('Loss/loss', value, epoch)
            elif name == 'avg_val_loss':
                self.writer_val.add_scalar('Loss/loss', value, epoch)
            elif name == 'avg_dice':
                self.writer_val.add_scalar('Dice/avg', value, epoch)
            elif name.startswith('dice_'):
                self.writer_val.add_scalar('Dice/' + name[len('dice_'):], value, epoch)
            elif name.startswith('avg_'):
                self.writer_val.add_scalar('Metrics/' + name[len('avg_'):], value, epoch)
            else:
                self.writer_val.add_scalar('Metrics/' + name, value, epoch)

    def on_completed(self, engine):
        if self.writer_train:
            self.writer_train.close()
        if self.writer_val:
            self.writer_val.close()