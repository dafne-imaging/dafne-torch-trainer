from dataclasses import asdict
import torch.nn as nn

from ..config.config_params import ModelConfig, TrainingConfig

class DafneModelWrapper(nn.Module):
    '''
    Wrapper class for the core model.
    '''
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.model = core_model
        self.ewc_data = None

    def forward(self, x):
        output = self.model(x)

        if isinstance(output, (list, tuple)):
            return output[0]
        return output
    
    def load_weights(self, pretrained_dict: dict):
        '''
        Load weights from a pretrained model, skipping layers that don't match.
        '''
        model_dict = self.model.state_dict()
        compatible_weights = {}
        
        skipped_layers = []

        for name, param in pretrained_dict.items():
            if name in model_dict:
                if param.shape == model_dict[name].shape:
                    compatible_weights[name] = param
                else:
                    skipped_layers.append(f'{name} (not same shape) \
                        {param.shape} vs {model_dict[name].shape}')

            else: 
                skipped_layers.append(f'{name} not in the new model')
        
        model_dict.update(compatible_weights)
        self.model.load_state_dict(model_dict)
    
    def get_state_dict(self, merge_lora=True):
        if merge_lora and hasattr(self.model, 'get_merged_model'):
            merged = self.model.get_merged_model()
            return merged.state_dict()
        return self.model.state_dict()
    
    def save_model_and_metadata(self, 
                                model_config: ModelConfig,
                                training_config: TrainingConfig, 
                                save_path: str):
        
        from ..bin.create_torch_model import create_dynamic_model


        net_metadata = {**asdict(model_config), **model_config.extra_params}
        train_metadata = asdict(training_config)
        
        if save_path.endswith('.pth'):
            save_path = save_path.replace('.pth', '.model')
        elif not save_path.endswith('.model'):
            save_path += '.model'
            
        with open(save_path, 'wb') as f:
            create_dynamic_model(weights=self.get_state_dict(), 
                                net_metadata=net_metadata, 
                                train_metadata=train_metadata,
                                ewc_data=self.ewc_data).dump(f)
