import torch.nn as nn
import monai.networks.nets as nets
from ..config.config_params import ModelConfig


class ModelFactory:

    _REGISTRY = {
        'unet': nets.Unet,
        'dynunet': nets.DynUnet,
        # add here other segmentation models
    }

    @staticmethod
    def create_model(config: ModelConfig)->nn.Module:
        model_class = ModelFactory._REGISTRY.get(config.model_name)
        if model_class is None:
            raise ValueError(f"Model {config.model_name} not found")
        
        # common parameters in all models
        params = {
            'in_channels': config.in_channels,
            'spatial_dims': config.spatial_dims,
            'out_channels': config.n_classes,
        }

        full_params = {**params, **config.extra_params}
        
        return model_class(**full_params)