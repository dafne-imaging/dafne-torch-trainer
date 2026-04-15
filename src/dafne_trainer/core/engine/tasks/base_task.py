from abc import ABC, abstractmethod
from typing import Any
from ..trainer_engine import Engine

class BaseTask(ABC):
    """
    Interface for training and validation tasks.
    A Task defines how to process a single batch.
    """

    @abstractmethod
    def train_step(self, engine: Engine, batch: Any) -> Any:
        """
        Process a single batch during training.

        Args:
            engine: The current Engine instance.
            batch: The current data batch.
        
        Returns:
            The value to be stored in engine.state.output (usually loss or metrics).
        """
        pass

    @abstractmethod
    def validation_step(self, engine: Engine, batch: Any) -> Any:
        """
        Process a single batch during validation.
        
        Args:
            engine: The current Engine instance.
            batch: The current data batch.
        
        Returns:
            The batch results to be stored in engine.state.output (usually preds, targets).
        """
        pass