import sys 
import os
import torch
import numpy as np
import dill
import argparse
import json
from collections import OrderedDict

from dafne_dl.DynamicTorchModel import DynamicTorchModel


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
    from dafne_models.core.transform.custom_transforms import PreprocessAnisotropy

    if not input_image.shape[0] < input_image.shape[1]: 
        input_image = np.ascontiguousarray(np.moveaxis(input_image, -1, 0))

    data = {'image': input_image}
    
    # Access metadata stored in the model object
    net_metadata = model_obj.metadata['net_metadata']
    dyn_model = net_metadata['use_dynamic']
    spacing = net_metadata['median_spacing']
    spatial_dims = net_metadata['spatial_dims']

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
                spatial_size=net_metadata['patch_size'], \
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


def create_dynamic_model(weights, net_metadata, train_metadata):
    '''
    Create dynamic model
    
    :param weights: Model weights
    :param net_metadata: Model metadata
    :param train_metadata: Training metadata
    '''

    metadata = {
        'net_metadata': net_metadata,
        'train_metadata': train_metadata
    }

    clean_params = {}
    for k, v in net_metadata.items():
        if hasattr(v, 'tolist'): # Se è un array numpy, lo converte in lista
            clean_params[k] = v.tolist()
        else:
            clean_params[k] = v

    # BAKE METADATA INTO SOURCE (Must start at column 0 for exec)
    build_model_src = f"""
def build_model():
    from dafne_models.models.dafne_networks import DafneUnetModel, DafneDynUnetModel
    params = {repr(clean_params)}
    
    if params['use_dynamic']:
        return DafneDynUnet(
            spatial_dims=params['spatial_dims'],
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            kernel_size=params['kernels'],
            strides=params['strides'],
            norm_name=("INSTANCE", {{"affine": True}}),
            deep_supervision=False
        )
    else:
        return DafneUnetModel(
            spatial_dims=params['spatial_dims'],
            in_channels=params['in_channels'],
            n_levels=params['n_levels'],
            kernel_size=params['kernel_size'],
            out_channels=params['out_channels']
        )
"""
    exec_scope = {}
    exec(build_model_src, globals(), exec_scope)
    build_model_baked = exec_scope['build_model']
    build_model_baked.source = build_model_src

    dynamic_model = DynamicTorchModel(
        model_id="Dafne_Custom_Model",
        init_model_function=build_model_baked,
        apply_model_function=apply_network_inf,
        weights=weights,  
        metadata=metadata,
        data_dimensionality=metadata['net_metadata']['spatial_dims']
    )

    return dynamic_model