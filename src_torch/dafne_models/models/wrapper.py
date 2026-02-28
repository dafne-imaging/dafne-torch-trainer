import torch.nn as nn

class DafneModelWrapper(nn.Module):
    '''
    Wrapper class for the core model.
    '''
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.model = core_model

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
    
