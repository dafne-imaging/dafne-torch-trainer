import numpy as np
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    ToTensord,
    CastToTyped,
    SpatialPadd,
    RandRotate90d,
    RandFlipd,
    RandZoomd, 
    RandGaussianNoised,
    RandCropByPosNegLabeld,
    DivisiblePadd
    )

from ..utils.optimizer import get_optimal_hyperparameters
from .dafne_dataset import MapTransformLoadData
from .transforms_utils import PreprocessAnisotropy


def build_transform_list(keys:list,
                         median_spacing:list, 
                         train_transforms:bool,
                         augm_params:dict,
                         spatial_dims:int=2) -> list:
    
    if spatial_dims == 3: 
        pipeline = [
            MapTransformLoadData(keys=keys, spatial_dims=3),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image', 'mask'], 
                                                    target_spacing=median_spacing,
                                                    model_mode='train' if train_transforms else None)
        ]

        if train_transforms:
            
            pipeline.append( 
                RandCropByPosNegLabeld(
                    keys=['image', 'mask'], label_key='mask',
                    spatial_size=(16, 96, 96), # Patch 3D
                    pos=1, neg=1, num_samples=4,
                    image_key='image', image_threshold=0
                )
            )
                
            if augm_params.get('rotate'):
                pipeline.append(RandRotate90d(keys=['image', 'mask'], prob=0.5, spatial_axes=(1,2)))
            if augm_params.get('flip_x'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1))
            if augm_params.get('flip_y'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2))
            if augm_params.get('zoom'):
                pipeline.append(RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=['bilinear', 'nearest']))
            if augm_params.get('noise'):
                pipeline.append(RandGaussianNoised(keys=['image'], prob=0.5, std=0.05))
    
            pipeline.append(ToTensord(keys=['image', 'mask']))
            
            if not train_transforms:
                pipeline.append(DivisiblePadd(keys=['image', 'mask'], k=32, mode='edge'))
    
    else: 
        pipeline = [
            MapTransformLoadData(keys=keys, spatial_dims=2),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image', 'mask'], 
                                                target_spacing=median_spacing,
                                                model_mode='train' if train_transforms else None)
        ]
    
        if train_transforms:
            if augm_params.get('rotate'):
                pipeline.append(RandRotate90d(keys=['image', 'mask'], prob=0.5, spatial_axes=(0,1)))
            if augm_params.get('flip_x'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0))
            if augm_params.get('flip_y'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1))
            if augm_params.get('zoom'):
                pipeline.append(RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=['bilinear', 'nearest']))
            if augm_params.get('noise'):
                pipeline.append(RandGaussianNoised(keys=['image'], prob=0.5, std=0.05))
            
    pipeline.extend([ToTensord(keys=['image', 'mask']), 
                        DivisiblePadd(keys=['image', 'mask'], k=32)])
    
    return pipeline


def build_transforms_dynunet(keys: list,
                             patch_size: list, 
                             target_spacing: list,
                             train_transforms: bool = True,
                             augm_params: dict = None
                            )-> list:

    pipeline = [
        MapTransformLoadData(keys=['filepath'], spatial_dims=3),
        EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
        PreprocessAnisotropy(keys=['image', 'mask'], 
                                                target_spacing=target_spacing,
                                                model_mode='train' if train_transforms else None,
                                                spatial_dims=3)
        ]
    pipeline.append(
        SpatialPadd(keys=keys, spatial_size=patch_size, method="symmetric")
    )
    
    if train_transforms:
        pipeline.append(
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="mask",
                spatial_size=patch_size,
                pos=1, neg=1, num_samples=1, 
                image_key="image", image_threshold=0,
            )
        )
        
        # Augmentation (Parametriche)
        if augm_params.get('flip_x'):
            pipeline.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=0))
        if augm_params.get('flip_y'):
            pipeline.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=1))
        if augm_params.get('rotate'):
            pipeline.append(RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)))
        if augm_params.get('zoom'):
            pipeline.append(RandZoomd(keys=keys, prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=("bilinear", "nearest")))
        if augm_params.get('noise'):
            pipeline.append(RandGaussianNoised(keys=["image"], prob=0.2, std=0.05))

    if not train_transforms:
        pipeline.append(
            SpatialPadd(keys=keys, spatial_size=patch_size, method="symmetric")
        )

    pipeline.extend([
        CastToTyped(keys=keys, dtype=(np.float32, np.uint8)),
        ToTensord(keys=keys)
    ])

    return pipeline



