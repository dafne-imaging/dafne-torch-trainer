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

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, received: {self.r}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), received: {self.lora_dropout}")
        if self.rank_for not in ("channels", "spatial"):
            raise ValueError(f"rank_for must be 'channels' or 'spatial', received: {self.rank_for!r}")


@dataclass
class DatasetConfig:
    root_dir: Optional[Path] = None
    val_split: float = 0.2
    random_seed: int = 42
    
    target_spacing: Optional[tuple] = None
    median_shape: Optional[tuple] = None

    def __post_init__(self):
        if not 0 < self.val_split < 1:
            raise ValueError(f"val_split must be in (0, 1), received: {self.val_split}")


@dataclass
class ModelConfig:

    # model parameters
    model_name: str = 'unet'
    spatial_dims: int = 3
    #dimensionality
    
    out_channels: int = 2
    in_channels: int = 1
    use_dynamic: bool = False
    patch_size: tuple = None
    median_shape: tuple = None
    median_spacing: tuple = None
    n_levels: int = 5
    labels_name: List[str] = None
    #model_type
    
    # specific model parameters for monai model
    extra_params: dict = field(default_factory=dict)

    # fine-tuning parameters
    fine_tuning: bool = False
    percent_to_freeze: float = 0.0
    gradual_unfreezing: bool = False

    # lora parameters
    lora_config: Optional[LoraConfig] = None

    def __post_init__(self):
        if self.spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, received: {self.spatial_dims}")
        if self.percent_to_freeze is not None and not 0 <= self.percent_to_freeze <= 1:
            raise ValueError(f"percent_to_freeze must be in [0, 1], received: {self.percent_to_freeze}")

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

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, received: {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, received: {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, received: {self.learning_rate}")


@dataclass
class InferenceMetricsConfig:
    '''
    Configuration class for inference metrics.
    '''
    include_background: bool = False
    reduction: str = "mean_batch"

    compute_dice: bool = True
    compute_hausdorff_95: bool = True
    compute_surface_distance: bool = True
    compute_precision: bool = True
    compute_sensitivity: bool = True
    compute_specificity: bool = True
    compute_accuracy: bool = True
    compute_f1_score: bool = True
    compute_threat_score: bool = True

    hd_percentile: float = 95.0
    sd_percentile: float = 95.0
