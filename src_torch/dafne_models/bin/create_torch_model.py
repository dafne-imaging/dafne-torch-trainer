import sys 
import os
import torch
import numpy as np
import dill
import numpy
import argparse
import json
from collections import OrderedDict

from dafne_dl.DynamicTorchModel import DynamicTorchModel

CONFIG = {
    'SPATIAL_DIMS': 3,
    'N_LEVELS': 5,
    'IN_CHANNELS': 1,
    'N_CLASSES': 2,
    'KERNEL_SIZE': 3,
    'OUT_CHANNELS':2,
    'MEDIAN_SPACING': [1.0, 1.0, 1.0],
    'USE_DYNAMIC' : False, 
    'KERNELS' : [],
    'STRIDES' : [],
}


def init_network():
    '''
    Inizialize dafne unet network 
    '''
    from ..models.dafne_networks import DafneUnetModel, DafneDynUnet

    if CONFIG['USE_DYNAMIC']:
        # Ricostruzione DynUNet
        model = DafneDynUnet(
            in_channels=CONFIG['IN_CHANNELS'],
            out_channels=CONFIG['N_CLASSES'],
            kernel_size=CONFIG['KERNELS'],
            strides=CONFIG['STRIDES']
        )
    else:
        model = DafneUnetModel(
            spatial_dims=CONFIG['SPATIAL_DIMS'],
            in_channels=CONFIG['IN_CHANNELS'],
            n_levels=CONFIG['N_LEVELS'],
            kernel_size=CONFIG['KERNEL_SIZE'],
            out_channels=CONFIG['OUT_CHANNELS']
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
        PreprocessAnisotropy(keys=['image'], target_spacing=CONFIG['MEDIAN_SPACING'],
                            model_mode=None, spatial_dims=CONFIG['SPATIAL_DIMS']),
        DivisiblePadd(keys=['image'], k=32),
        ToTensord(keys=['image'])
        ]

    data_processed = Compose(transf_list)(data)
    
    img_tensor = data_processed['image']

    model_obj.model.eval()
    with torch.no_grad():
        if CONFIG['SPATIAL_DIMS'] == 3: 
            img_tensor = img_tensor.unsqueeze(0).to(model_obj.device)
            output = model_obj.model(img_tensor)
            pred_torch = torch.argmax(output, dim=1)
            pred_vol = pred_torch[0].detach().cpu().numpy().astype(np.int8)

        elif CONFIG['SPATIAL_DIMS'] == 2: 
            pred_vol = []
            depth = img_tensor.shape[1]
            for i in range(depth):
                slice_torch = img_tensor[:, i, :, :].unsqueeze(0).to(model_obj.device)
                output = model_obj.model(slice_torch)
                pred_torch = torch.argmax(output, dim=1)
                pred_vol.append(pred_torch[0].detach().cpu().numpy().astype(np.int8))
            
            pred_vol = np.stack(pred_vol, axis=0)
        
        return pred_vol


def create_dynamic_model(weights, net_metadata, train_metadata):
    '''
    Create dynamic model
    
    :param model: Model object
    :param weights: Model weights
    :param net_metadata: Model metadata
    :param train_metadata: Training metadata
    '''

    metadata = {
        'net_metadata': net_metadata,
        'train_metadata': train_metadata
    }

    def build_model():
        from ..models.dafne_networks import DafneUnetModel, DafneDynUnet

        if metadata['net_metadata']['use_dynamic']:
            return DafneDynUnet(
                spatial_dims=metadata['net_metadata']['spatial_dims'],
                in_channels=metadata['net_metadata']['in_channels'],
                out_channels=metadata['net_metadata']['out_channels'],
                kernel_size=metadata['net_metadata']['kernels'],
                strides=metadata['net_metadata']['strides'],
                norm_name=("INSTANCE", {"affine": True}),
                deep_supervision=False
            )
        else:
            return DafneUnetModel(
                spatial_dims=metadata['net_metadata']['spatial_dims'],
                in_channels=metadata['net_metadata']['in_channels'],
                n_levels=metadata['net_metadata']['n_levels'],
                kernel_size=metadata['net_metadata']['kernel_size'],
                out_channels=metadata['net_metadata']['out_channels']
            )
    
    def apply_network_inf(model_obj, input_image):
        import numpy as np
        import torch
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            ToTensord,
            SpatialPadd, 
            CastToTyped,
            DivisiblePadd
        )
        from dafne_models.core.transforms_utils import PreprocessAnisotropy

        if not input_image.shape[0] < input_image.shape[1]: 
            input_image = np.ascontiguousarray(np.moveaxis(input_image, -1, 0))

        dyn_model = model_obj.metadata['net_metadata']['use_dynamic']
        data = {'image': input_image}
        spacing = model_obj.metadata['net_metadata']['median_spacing']
        spatial_dims = model_obj.metadata['net_metadata']['spatial_dims']

        if not dyn_model:
            transf_list = [
                EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
                PreprocessAnisotropy(keys=['image'], target_spacing=spacing,
                                    model_mode=None, spatial_dims=spatial_dims),
                DivisiblePadd(keys=['image'], k=32),
                ToTensord(keys=['image'])
            ]
        else: 
            transf_list = [
                EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
                PreprocessAnisotropy(keys=['image'], target_spacing=spacing,
                                    model_mode=None, spatial_dims=spatial_dims),
                SpatialPadd(keys=['image'], \
                    spatial_size=model_obj.metadata['net_metadata']['patch_size'], \
                    method="symmetric"),
                CastToTyped(keys=['image'], dtype=np.float32),
                ToTensord(keys=['image'])
            ]

        data_processed = Compose(transf_list)(data)
        img_tensor = data_processed['image']
        
        model_obj.model.eval()
        with torch.no_grad():
            if spatial_dims == 3: 
                img_tensor = img_tensor.unsqueeze(0).to(model_obj.device)
                output = model_obj.model(img_tensor)
                pred_torch = torch.argmax(output, dim=1)
                pred_vol = pred_torch[0].detach().cpu().numpy().astype(np.int8)

            elif spatial_dims == 2: 
                pred_vol = []
                depth = img_tensor.shape[1]
                for i in range(depth):
                    slice_torch = img_tensor[:, i, :, :].unsqueeze(0).to(model_obj.device)
                    output = model_obj.model(slice_torch)
                    pred_torch = torch.argmax(output, dim=1)
                    pred_vol.append(pred_torch[0].detach().cpu().numpy().astype(np.int8))
            
                pred_vol = np.stack(pred_vol, axis=0)
        
        return pred_vol
    

    dynamic_model = DynamicTorchModel(
        model_id="Dafne_Custom_Model",
        init_model_function=build_model,
        apply_model_function=apply_network_inf,
        weights=weights,  
        metadata=metadata,
        data_dimensionality=metadata['net_metadata']['spatial_dims']
    )

    return dynamic_model


def main():
    parser = argparse.ArgumentParser(description='Pack trained model into .dafne format')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing _params.json and .pth weights')
    parser.add_argument('--weights_name', type=str, default='_best_model.pth', help='Name of the weights file inside model_dir')
    parser.add_argument('--output', type=str, required=True, help='Output filename (e.g., my_model.dafne)')
    parser.add_argument('--metadata', type=dict, default={}, required=True, help='Network and training metadata')

    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, '_params.json')
    if not os.path.exists(json_path):
        print(f"Error: Params file not found at {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    print("Loaded params:", json.dumps(params, indent=2))

    CONFIG['SPATIAL_DIMS'] = params.get('spatial_dims', 3)
    CONFIG['N_LEVELS'] = params.get('n_levels', 5)
    CONFIG['IN_CHANNELS'] = params.get('in_channels', 1)
    CONFIG['N_CLASSES'] = params.get('out_channels', 2)
    CONFIG['KERNEL_SIZE'] = params.get('kernel_size', 3)
    CONFIG['MEDIAN_SPACING'] = params.get('median_spacing', [1.0, 1.0, 1.0])
    CONFIG['USE_DYNAMIC'] = params.get('use_dynamic', False)
    CONFIG['KERNELS'] = params.get('kernels', [])
    CONFIG['STRIDES'] = params.get('strides', [])

    print('Load model weights...')
    if not os.path.exists(os.path.join(args.model_dir, args.weights_name)):
        try:
            pth_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pth')]
            print(f"Warning: '{args.weights_name}' not found. Using '{pth_files[0]}' instead.")
            args.weights_name = pth_files[0]
        except:
            print(f"Error: No weights file found in {args.model_dir}")
            sys.exit(1)
    else:
        print(f'Load model weights from: {os.path.join(args.model_dir, args.weights_name)}')
        state_dict = torch.load(os.path.join(args.model_dir, args.weights_name), map_location='cpu')
    
    # DynamicTorchModel in dafne-dl repo
    dynamic_model = DynamicTorchModel(
        model_id="Dafne_Custom_Model",
        init_model_function=init_network,
        apply_model_function=apply_network,
        weights=state_dict,  
        data_dimensionality=CONFIG['SPATIAL_DIMS']
    )
    
    # dump dynamic model
    try: 
        print(f"Dumping model to {args.output}")
        with open(args.output, 'wb') as file:
            dynamic_model.dump(file)
            print(f"Model dumped in {args.output} successfully!")
    except Exception as e: 
        print("Error during model dump: {e}")

if __name__ == '__main__':    
    main()