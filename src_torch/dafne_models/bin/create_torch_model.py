import sys
import os
import torch
import numpy as np
import dill
import argparse
import json
import uuid
from collections import OrderedDict

from dafne_dl.DynamicTorchModel import DynamicTorchModel


def apply_network_inf(model_obj, data_dict: dict) -> dict:
    from dafne_inference.inference import run_inference
    return run_inference(model_obj, data_dict)


def create_dynamic_model(weights, net_metadata, train_metadata):
    '''
    Create dynamic model
    
    :param weights: Model weights
    :param net_metadata: Model metadata
    :param train_metadata: Training metadata
    '''

    metadata = {
        'net_metadata': net_metadata,
        'train_metadata': train_metadata,
        'dependencies': {
            'dafne_inference': 'dafne-monai-inference>=0.1.0'
        }
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
    from dafne_inference.networks import DafneUnetModel, DafneDynUnetModel
    params = {repr(clean_params)}
    
    if params['use_dynamic']:
        return DafneDynUnetModel(
            spatial_dims=params['spatial_dims'],
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            kernels=params['kernels'],
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
        model_id=uuid.uuid4(),
        init_model_function=build_model_baked,
        apply_model_function=apply_network_inf,
        weights=weights,  
        metadata=metadata,
        data_dimensionality=metadata['net_metadata']['spatial_dims']
    )

    return dynamic_model