from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class DatasetConfig:
    root_dir: Optional[Path] = None
    file_list: List[Path] = field(default_factory=list)
    val_split: float = 0.2
    random_seed: int = 42
    
    target_spacing: Optional[tuple] = None
    median_shape: Optional[tuple] = None


@dataclass
class ModelConfig:

    # model parameters
    model_name: str = 'unet'
    spatial_dims: int = 3
    n_classes: int = 2
    in_channels: int = 1
    use_dynamic: bool = False
    patch_size: tuple = (16, 96, 96)
    
    # specific model parameters for monai model
    extra_params: dict = field(default_factory=dict)

@dataclass
class AugmentationConfig:
    rotate: bool = False
    flip: bool = False
    zoom: bool = False
    shift: bool = False


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 2
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    mixed_precision: bool = False
    scheduler: bool = False