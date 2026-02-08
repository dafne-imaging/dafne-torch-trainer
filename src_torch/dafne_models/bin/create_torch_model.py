import sys 
import os
import torch
import numpy as np
import dill
import numpy
from collections import OrderedDict

from dafne_dl.DynamicTorchModel import DynamicTorchModel

SPATIAL_DIMS = 3
N_LEVELS = 5
IN_CHANNELS = 1
N_CLASSES = 4
KERNEL_SIZE = 3
MEDIAN_SPACING = [8.0, 1.4, 1.4] #example value

WEIGHT_PATH = ''
OUTPUT_FILENAME = ''


def init_network():
    '''
    Inizialize dafne unet network 
    '''
    from dafne_models.models.dafne_network import DafneUnetModel

    model = DafneUnetModel(
        spatial_dims=SPATIAL_DIMS,
        in_channels=IN_CHANNELS,
        n_levels=N_LEVELS,
        kernel_size=KERNEL_SIZE,
        out_channels=N_CLASSES
    )

    return model


def apply_network(model_obj, input_image):
    '''
    Apply network on a single volumes
    
    :param model_obj: Descrizione
    :param input_image: Descrizione
    '''
    import numpy as np
    from monai.data import DataLoader
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        ToTensord,
        DivisiblePadd
    )

    from dafne_models.core.transforms_utils import PreprocessAnisotropy

    if not input_image.shape[0] < input_image.shape[1]: 
        input_image = np.ascontiguousarray(np.moveaxis(input_image, -1, 0))

    data = {'image': input_image}

    transf_list = [
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        PreprocessAnisotropy(keys=['image'], target_spacing=MEDIAN_SPACING,
                            model_mode=None, spatial_dims=SPATIAL_DIMS),
        DivisiblePadd(keys=['image'], k=32),
        ToTensord(keys=['image'])
        ]

    data_processed = Compose(transf_list)(data)
    
    img_tensor = data_processed['image']

    model_obj.model.eval()
    with torch.no_grad():
        if SPATIAL_DIMS == 3: 
            img_tensor = img_tensor.unsqueeze(0).to(model_obj.device)
            output = model_obj.model(img_tensor)
            pred_torch = torch.argmax(output, dim=1)
            pred_vol = pred_torch[0].detach().cpu().numpy().astype(np.int8)

        elif SPATIAL_DIMS == 2: 
            pred_vol = []
            depth = img_tensor.shape[1]
            for i in range(depth):
                slice_torch = img_tensor[:, i, :, :].unsqueeze(0).to(model_obj.device)
                output = model_obj.model(slice_torch)
                pred_torch = torch.argmax(output, dim=1)
                pred_vol.append(pred_torch[0].detach().cpu().numpy().astype(np.int8))
            
            pred_vol = np.stack(pred_vol, axis=0)
        
        return pred_vol


def main():

    print('Load model weights...')
    if not os.path.exists(WEIGHT_PATH):
        print(f'Error: Weights file not found!')
        sys.exit(1)
    else:
        state_dict = torch.load(WEIGHT_PATH, map_location='cpu')
    
    # define here DynamicTorchModel
    
    dynamic_model = DynamicTorchModel(
        model_id="Dafne_Custom_Model",
        init_model_function=init_network,
        apply_model_function=apply_network,
        weights=state_dict,  
        data_dimensionality=SPATIAL_DIMS
    )
    
    print(f"Dumping model to {OUTPUT_FILENAME}...")
    # 3. Salva il file .dafne (Codice + Pesi)
    with open(OUTPUT_FILENAME, 'wb') as f:
        dynamic_model.dump(f)

if __name__ == '__main__':
    main()