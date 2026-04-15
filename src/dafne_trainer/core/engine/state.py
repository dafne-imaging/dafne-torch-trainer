from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import torch

@dataclass
class EngineState:
    '''
    State class that contain training information
    '''

    #Progress information
    epoch: int = 0
    iteration: int = 0
    max_epochs: int = 0

    #Data information
    batch: torch.Tensor = None
    output: torch.Tensor = None
    loss: float = 0.0

    #Metrics and history
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_metric: float = None

    #Metadata
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    # prediction on random validation images
    # here to save random validation images and their predictions
    predictions: Dict[str, Any] = field(default_factory=dict)

    pred_dice_per_mask: Dict[str, float] = field(default_factory=dict)

    #Training information
    terminated: bool = False
    device: Optional[torch.device] = None
    check_stop: Optional[Callable] = None

    def update_metrics(self, new_metric:Dict[str, float]):
        self.metrics.update(new_metric)
    
    def update_history(self, metrics:List[Dict[str, Any]]):
        self.history.append(metrics)