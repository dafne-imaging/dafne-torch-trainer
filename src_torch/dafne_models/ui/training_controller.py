"""
Training controller: builds config objects from raw UI values.
"""

from ..config.config_params import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    AugmentationConfig,
    LoraConfig,
    InferenceMetricsConfig,
)


def build_training_configs(
    # paths
    data_path: str,
    pretrained_path: str,
    # model architecture
    n_levels: int,
    kernel_size: int,
    conv_layers: int,
    is_3d: bool,
    use_dynunet: bool,
    # training
    epochs: int,
    batch_size: int,
    lr: float,
    early_stopping: bool,
    mixed_precision: bool,
    scheduler: bool,
    # augmentation and adaptation (still plain dicts from the dialogs)
    augm_params: dict,
    adaptation_params: dict,
) -> dict:
    """
    Receives raw values from the UI and returns ready-to-use config objects.
    Returns a dict with keys: dataset_config, model_config, train_config, inference_config.
    """

    dataset_config = DatasetConfig(
        root_dir=data_path,
        val_split=0.2,
        random_seed=42,
    )

    model_config = ModelConfig(
        model_name='dynunet' if use_dynunet else 'unet',
        spatial_dims=3 if is_3d else 2,
        out_channels=2,
        in_channels=1,
        use_dynamic=use_dynunet,
        extra_params={
            'n_levels': n_levels,
            'kernel_size': kernel_size,
            'num_res_units': conv_layers,
            'kernels': None,
            'strides': None,
        },
    )

    augm_config = AugmentationConfig(
        rotate=augm_params.get('rotate', False),
        flip_x=augm_params.get('flip_x', False),
        flip_y=augm_params.get('flip_y', False),
        zoom=augm_params.get('zoom', False),
        noise=augm_params.get('noise', False),
    )

    train_config = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        augmentation=augm_config,
        pretrained_model_path=pretrained_path,
        mixed_precision=mixed_precision,
        early_stopping=early_stopping,
        scheduler=scheduler,
    )

    inference_config = InferenceMetricsConfig(
        include_background=False,
        reduction='mean_batch',
    )

    # Apply adaptation mode to model_config
    mode = adaptation_params.get('mode', 'scratch')
    if mode == 'finetune':
        model_config.fine_tuning = True
        model_config.percent_to_freeze = adaptation_params.get('freeze_degree', 0.5)
        model_config.gradual_unfreezing = adaptation_params.get('gradual_unfreeze', False)
    elif mode == 'lora':
        model_config.fine_tuning = True
        model_config.percent_to_freeze = None
        model_config.lora_config = LoraConfig(
            r=adaptation_params.get('lora_rank', 8),
            lora_alpha=adaptation_params.get('lora_alpha', 16),
            lora_dropout=0.1,
            rank_for='channels',
            target_modules=['.*'],
        )
    else:  # scratch
        model_config.fine_tuning = False
        model_config.percent_to_freeze = None
        model_config.lora_config = None

    return {
        'dataset_config': dataset_config,
        'model_config': model_config,
        'train_config': train_config,
        'inference_config': inference_config,
    }