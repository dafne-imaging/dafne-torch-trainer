import logging
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


class QtSignalHandler(logging.Handler):
    """Forwards log records to a PyQt5 signal (message, levelname)."""

    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        try:
            msg = self.format(record)
            self.signal.emit(msg, record.levelname)
        except Exception:
            self.handleError(record)

from .train import run_training

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
    
    # Send data to cpu (float, numpy, numpy, float, numpy, float, dict)
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

    # Log message (text, levelname) forwarded to the GUI log console
    sig_log = pyqtSignal(str, str)

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

    def stop(self):
        self.is_running = False      

    def run(self):
        handler = QtSignalHandler(self.sig_log)
        handler.setFormatter(logging.Formatter('%(name)s — %(message)s'))
        logger = logging.getLogger('dafne_models')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            run_training(self.model_config, self.dataset_config,
                         self.inference_metrics,
                         self.train_config,
                         self.save_path,
                         self.sig_status,
                         self.sig_progress,
                         self._callback_log,
                         self._callback_check_stop,
                         self.sig_update_plot,
                         self.sig_error,
                         self.sig_stopped,
                         self.sig_finished,
                         self.device)
        finally:
            logger.removeHandler(handler)