from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    rank_for: str = "channels"
    target_modules: Optional[List[str]] = None


@dataclass
class DatasetConfig:
    root_dir: Optional[Path] = None
    val_split: float = 0.2
    random_seed: int = 42
    
    target_spacing: Optional[tuple] = None
    median_shape: Optional[tuple] = None


@dataclass
class ModelConfig:

    # model parameters
    model_name: str = 'unet'
    spatial_dims: int = 3
    out_channels: int = 2
    in_channels: int = 1
    use_dynamic: bool = False
    patch_size: tuple = None
    median_shape: tuple = None
    median_spacing: tuple = None
    n_levels: int = 5
    
    # specific model parameters for monai model
    extra_params: dict = field(default_factory=dict)

    # fine-tuning parameters
    fine_tuning: bool = False
    percent_to_freeze: float = 0.0

    # lora parameters
    lora_config: Optional[LoraConfig] = None

@dataclass
class AugmentationConfig:
    rotate: bool = False
    flip_x: bool = False
    flip_y: bool = False
    zoom: bool = False
    noise: bool = False
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
    pretrained_model_path: Optional[Path] = None
    mixed_precision: bool = False
    early_stopping: bool = False
    scheduler: bool = False
    mixed_precision: bool = False