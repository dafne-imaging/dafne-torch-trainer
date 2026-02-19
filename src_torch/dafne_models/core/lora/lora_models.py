import re 
from torch import nn, Tensor
from .layers import LoRA2dConvLayer, LoRA3dConvLayer, LoRANdConvLayer

class LoRAModel(nn.Module):
    def __init__(self, base_model:nn.Module, lora_config:dict) -> None: 
        super(LoRAModel, self).__init__()

        self.base_model = base_model
        
        assert isinstance(self.base_model, nn.Module), 'Invalid type! The base model should be a torch.nn.Module'

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False #freeze base model parameters

        self.lora_module_names = []
        
        for target_module_name, config in lora_config.items():
            for module_name, module in self.base_model.named_modules():
                if re.match(target_module_name, module_name):
                    self.lora_module_names.append(module_name)
                    if isinstance(module, nn.modules.conv._ConvNd):
                        lora_module = LoRANdConvLayer(module, config)
                    else: 
                        raise AssertionError('Invalid Target Module Type! Only Conv layers are supported')
                    setattr(self.base_model, module_name, lora_module)

    def enable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).enable_adapter()
    
    def disable_adapter(self) -> None:
        for lora_module_name in self.lora_module_names:
            getattr(self.base_model, lora_module_name).disable_adapter()
    
    def forward(self, x:Tensor) -> Tensor:
        return self.base_model(x)
    
    def get_merged_model(self) -> nn.Module:
        merged_model = self.base_model.__class__()
        for module_name, module in merged_model.named_modules():
            if module_name == '':
                continue
            
            if module_name in self.lora_module_names:
                setattr(merged_model, module_name, getattr(self.base_model, module_name).get_merged_modules())
            else:
                setattr(merged_model, module_name, getattr(self.base_model, module_name))
            
        return merged_model
                