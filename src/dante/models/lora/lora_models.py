import re 
import copy

from torch import nn, Tensor
from .layers import LoRA2dConvLayer, LoRA3dConvLayer, LoRANdConvLayer, LoRALinearLayer

class LoRAModel(nn.Module):
    '''
        LoRA wrapper model
    '''
    def __init__(self, base_model:nn.Module, lora_config:dict) -> None: 
        '''
        Args: 
            base_model: base model to wrap
            lora_config: dictionary that contain  
                the configuration for LoRA layers, including:
                - rank: the rank of the update matrices.
                - alpha: the scaling factor.
        '''
        super(LoRAModel, self).__init__()

        self.base_model = base_model
        
        assert isinstance(self.base_model, nn.Module), 'Invalid type! The base model should be a torch.nn.Module'

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False #freeze base model parameters

        self.lora_module_names = []
        
        for target_module_name, config in lora_config.items():
            for module_name, module in self.base_model.named_modules():
                if re.match(target_module_name, module_name):
                    if isinstance(module, nn.modules.conv._ConvNd):
                        self.lora_module_names.append(module_name)
                        lora_module = LoRANdConvLayer(module, config)
                        self._recursive_setattr(self.base_model, module_name, lora_module)
                    elif isinstance(module, nn.Linear):
                        self.lora_module_names.append(module_name)
                        lora_module = LoRALinearLayer(module, config)
                        self._recursive_setattr(self.base_model, module_name, lora_module)
                    else: 
                        pass

    def _recursive_setattr(self, obj, attr_path, value):
        parts = attr_path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def enable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            lora_module = self.base_model.get_submodule(lora_module_name)
            lora_module.enable_adapter()
    
    def disable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            lora_module = self.base_model.get_submodule(lora_module_name)
            lora_module.disable_adapter()
    
    def forward(self, x:Tensor) -> Tensor:
        return self.base_model(x)
    
    def get_merged_model(self) -> nn.Module:
        merged_model = copy.deepcopy(self.base_model)

        for module_name in self.lora_module_names:
            lora_module = self.base_model.get_submodule(module_name) #get submodule where lora layer is present
            merged_module = lora_module._get_merged_modules()
            self._recursive_setattr(merged_model, module_name, merged_module)
        
        return merged_model
                