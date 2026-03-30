import logging
import torch.nn as nn
import torch
import inspect

logger = logging.getLogger(__name__)
from .dafne_networks import DafneUnetModel, DafneDynUnetModel
from ..config.config_params import ModelConfig
from .lora.lora_models import LoRAModel
from dafne_dl.DynamicTorchModel import DynamicTorchModel


class ModelFactory:

    _REGISTRY = {
        'unet': DafneUnetModel,
        'dynunet': DafneDynUnetModel,
        # add here other segmentation models
    }

    @staticmethod
    def create_model(config: ModelConfig, 
                    dynamic_torch_model_obj: DynamicTorchModel = None) -> nn.Module:

        model_out_channels = None

        if config.fine_tuning:
            if dynamic_torch_model_obj is None:
                raise ValueError("DynamicTorchModel object is required for fine-tuning mode.")

            model = dynamic_torch_model_obj.model

            try:
                model_out_channels = dynamic_torch_model_obj.metadata['net_metadata'].get('out_channels')
            except (AttributeError, KeyError):
                model_out_channels = getattr(model, 'out_channels', None)

            if config.out_channels != model_out_channels and hasattr(model, 'update_output_channels'):
                model.update_output_channels(config.out_channels)
        
        else:
            model_class = ModelFactory._REGISTRY.get(config.model_name)
            if model_class is None:
                raise ValueError(f"Model {config.model_name} not found in registry.")
            
            params = {
                'in_channels': config.in_channels,
                'spatial_dims': config.spatial_dims,
                'out_channels': config.out_channels,
            }
            
            full_params = {**params, **config.extra_params}

            signature = inspect.signature(model_class.__init__)
            valid_keys = [p.name for p in signature.parameters.values() if p.name != 'self']
            filtered_params = {k: v for k, v in full_params.items() if k in valid_keys}
            model = model_class(**filtered_params)
            
        if config.lora_config:
            target_modules = config.lora_config.target_modules or [".*"]
            
            lora_map = { target: {
                'rank': config.lora_config.r,
                'alpha': config.lora_config.lora_alpha,
                'rank_for': config.lora_config.rank_for
            } for target in target_modules }
            
            model = LoRAModel(model, lora_map)
            model.enable_adapter()

            # If the output layer was replaced (n_classes changed), its base weights are
            # randomly initialized. Unfreeze them so they train fully instead of being
            # constrained to rank-r LoRA updates, which is insufficient for a random layer.
            if model_out_channels is not None and config.out_channels != model_out_channels:
                for lora_module_name in model.lora_module_names:
                    lora_module = model.base_model.get_submodule(lora_module_name)
                    if hasattr(lora_module, 'base_module'):
                        if getattr(lora_module.base_module, 'out_channels', None) == config.out_channels:
                            for param in lora_module.base_module.parameters():
                                param.requires_grad = True

        elif config.percent_to_freeze is not None:
            ModelFactory.freeze_layers(model, config.percent_to_freeze)
        
        return model

    @staticmethod
    def freeze_layers(model: nn.Module, degree: float) -> None:
        '''
        Freeze a percentage of the model layers
        '''
        trainable_modules = [
            (name, module)
            for name, module in model.named_modules()
            if len(list(module.parameters(recurse=False))) > 0
        ]

        num_modules = len(trainable_modules)
        num_to_freeze = int(num_modules * degree)

        logger.info("Fine-tuning: freezing %d/%d parameter blocks (%.0f%%)", num_to_freeze, num_modules, degree * 100)
        
        for i, (name, module) in enumerate(trainable_modules):
            is_norm = isinstance(module, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                nn.LayerNorm, nn.GroupNorm
            ))

            if is_norm:
                for param in module.parameters():
                    param.requires_grad = True
                continue

            freeze = (i < num_to_freeze)
            for param in module.parameters(recurse=False):
                param.requires_grad = not freeze
    
    @staticmethod
    def get_trainable_layers(model: nn.Module) -> list:
        '''
        Get the trainable layers
        '''
        modules_to_train = [m for m in model.modules() if any(p.requires_grad for p in m.parameters(recurse=False))]
        return modules_to_train
    
    @staticmethod
    def gradual_unfreeze(model:nn.Module, degree_to_unfreeze:float, optimizer:torch.optim.Optimizer):
        trainable_blocks = [
            module for module in model.modules()
            if len(list(module.parameters(recurse=False))) > 0
        ]
        reverse_trainable_module = trainable_blocks[::-1]

        num_m = len(reverse_trainable_module)
        num_to_unfreeze = int(num_m * degree_to_unfreeze)

        for i, module in enumerate(reverse_trainable_module):
            is_norm = isinstance(module, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                nn.LayerNorm, nn.GroupNorm
            ))

            if is_norm:
                for param in module.parameters():
                    param.requires_grad = True
                continue

            trainable = (i < num_to_unfreeze)
            for param in module.parameters(recurse=False):
                param.requires_grad = trainable
        
        current_trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer.param_groups[0]['params'] = current_trainable_params
        return optimizer
    
    @staticmethod
    def change_lr_for_layer(model: nn.Module, base_lr:float):
        module_to_train = ModelFactory.get_trainable_layers(model)
        
        num_m = len(module_to_train)
        param_group = []
        for i, m in enumerate(module_to_train):
            multiplier = 0.1 + (0.9 * (i / (num_m - 1 if num_m > 1 else 1)))
            train_params = [p for p in m.parameters(recurse=False) if p.requires_grad]
            if not train_params:
                continue
            param_group.append({
                'params': train_params,
                'lr': base_lr * multiplier
            })
        return param_group