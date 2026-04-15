import numpy as np
from dataclasses import asdict
from abc import ABC, abstractmethod

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

from ...config.config_params import ModelConfig, TrainingConfig
from dafne_dl.DynamicTorchModel import DynamicTorchModel
from ...utils.data_fingerprint import DatasetFingerprint
from ...utils.optimizer import get_optimal_hyperparameters
from .custom_transforms import MapTransformLoadData, PreprocessAnisotropy


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
                SpatialPadd(keys=['image', 'mask'], spatial_size=(32, 96, 96), method='symmetric')
            )
            pipeline.append(
                RandCropByPosNegLabeld(
                    keys=['image', 'mask'], label_key='mask',
                    spatial_size=(32, 96, 96), # Patch 3D
                    pos=3, neg=1, num_samples=4,
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
        else:
            pipeline.append(
                SpatialPadd(keys=['image', 'mask'], spatial_size=(32, 96, 96), method='symmetric')
            )

        pipeline.append(ToTensord(keys=['image', 'mask']))
    
    else: 
        pipeline = [
            MapTransformLoadData(keys=keys, spatial_dims=2),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            PreprocessAnisotropy(keys=['image', 'mask'], 
                                 target_spacing=median_spacing,
                                 model_mode='train' if train_transforms else None,
                                 spatial_dims=2)
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
    
    return Compose(pipeline)


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
    if train_transforms:
        # 1. Augmentation
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

        pipeline.append(
            SpatialPadd(keys=keys, spatial_size=patch_size, method="symmetric")
        )
        
        pipeline.append(
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="mask",
                spatial_size=patch_size,
                pos=3, neg=1, num_samples=4, 
                image_key="image", image_threshold=0,
            )
        )
    else:
        # In validation padding is required to avoid crashing sliding_window_inference or collate
        pipeline.append(
            SpatialPadd(keys=keys, spatial_size=patch_size, method="symmetric")
        )

    pipeline.extend([
        CastToTyped(keys=keys, dtype=(np.float32, np.uint8)),
        ToTensord(keys=keys)
    ])

    return Compose(pipeline)


class AbstractTransformBuilder(ABC):
    '''
        Abstract class for building transforms

        param:
            model_config: ModelConfig
            training_config: TrainingConfig
    '''
    def __init__(self, model_config: ModelConfig, 
                training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
    
    @abstractmethod
    def build_transforms(self):
        pass
    
    @abstractmethod
    def get_transforms_static_model(self):
        pass
    
    @abstractmethod
    def get_transforms_dynamic_model(self):
        pass


class TransformBuilderTraining(AbstractTransformBuilder):
    '''
    Build transforms for training and validation 
    for generic segmentation model and dynamic unet
    '''
    def __init__(self, model_config: ModelConfig, 
                training_config: TrainingConfig,
                data_fingerprint: DatasetFingerprint
                ):
        super().__init__(model_config, training_config)
        self.data_fingerprint = data_fingerprint
    
    def build_transforms(self):
        '''
        Build transforms for training and validation
        '''
        if self.model_config.use_dynamic and self.model_config.spatial_dims == 3:
            return self.get_transforms_dynamic_model()
        else:
            return self.get_transforms_static_model()
    
    def get_transforms_dynamic_model(self):
        '''
        Build transforms for dynamic unet
        '''
        median_spacing = self.data_fingerprint.data_spacing
        median_shape = self.data_fingerprint.data_shape

        axis_divisors = np.prod(self.model_config.extra_params.get('strides'), axis=0) if \
                self.model_config.extra_params.get('strides') is not None else None

        patch_size, batch_size = get_optimal_hyperparameters(
            median_shape, spatial_dims=self.model_config.spatial_dims, is_finetune=False,
            mixed_precision=self.training_config.mixed_precision,
            divisor=axis_divisors
        )
            
        kernels, strides = self.data_fingerprint.get_kernel_and_strides(patch_size)

        self.model_config.extra_params['kernels'] = kernels
        self.model_config.extra_params['strides'] = strides
        self.training_config.batch_size = batch_size
        self.model_config.patch_size = patch_size

        augm_params_dict = asdict(self.training_config.augmentation)

        train_transforms =  build_transforms_dynunet(
            keys=['image', 'mask'],
            patch_size=patch_size,
            target_spacing=median_spacing,
            train_transforms=True,
            augm_params=augm_params_dict
            )

        valid_transforms = build_transforms_dynunet(
            keys=['image', 'mask'],
            patch_size=patch_size,
            target_spacing=median_spacing,
            train_transforms=False, 
            augm_params=augm_params_dict
        )

        return train_transforms, valid_transforms
    
    def get_transforms_static_model(self):
        '''
        Build transforms for unet
        '''
            
        self.model_config.patch_size = (32, 96, 96) if self.model_config.spatial_dims == 3 else None
        median_spacing = self.data_fingerprint.data_spacing
        
        augm_params_dict = asdict(self.training_config.augmentation)
        
        train_transforms = build_transform_list(['filepath'],
                                            median_spacing,
                                            True,
                                            augm_params_dict,
                                            self.model_config.spatial_dims)

        valid_transforms = build_transform_list(['filepath'],
                                                median_spacing,
                                                False,
                                                augm_params_dict,
                                                self.model_config.spatial_dims)
        
        return train_transforms, valid_transforms


class TransformBuilderFineTuning(AbstractTransformBuilder):
    '''
    Build transforms for fine-tuning
    '''

    def __init__(self, model_config: ModelConfig, 
                training_config: TrainingConfig,
                ):
        super().__init__(model_config, training_config)

    def build_transforms(self):
        '''
        Build transforms for fine-tuning
        '''
        if self.model_config.use_dynamic and self.model_config.spatial_dims == 3:
            return self.get_transforms_dynamic_model()
        else:
            return self.get_transforms_static_model()
    
    def get_transforms_dynamic_model(self):
        '''
        Build transforms for dynamic unet
        '''
        axis_divisors = np.prod(self.model_config.extra_params['strides'], axis=0) if \
                self.model_config.extra_params['strides'] is not None else None

        # In fine-tuning, we use the patch_size inherited from the model.
        # We only use get_optimal_hyperparameters to find the best batch_size for it.
        _, batch_size = get_optimal_hyperparameters(
            self.model_config.patch_size, self.model_config.spatial_dims, is_finetune=True,
            divisor=axis_divisors, mixed_precision=self.training_config.mixed_precision
        )

        patch_size = self.model_config.patch_size
        self.training_config.batch_size = batch_size
        augm_params_dict = asdict(self.training_config.augmentation)

        train_transforms =  build_transforms_dynunet(
            keys=['image', 'mask'],
            patch_size=patch_size,
            target_spacing=self.model_config.median_spacing,
            train_transforms=True,
            augm_params=augm_params_dict
            )

        valid_transforms = build_transforms_dynunet(
            keys=['image', 'mask'],
            patch_size=patch_size,
            target_spacing=self.model_config.median_spacing,
            train_transforms=False, 
            augm_params=augm_params_dict
        )

        return train_transforms, valid_transforms
    
    def get_transforms_static_model(self):
        '''
        Build transforms for unet
        '''
        augm_params_dict = asdict(self.training_config.augmentation)
        
        train_transforms = build_transform_list(['filepath'],
                                            self.model_config.median_spacing,
                                            True,
                                            augm_params_dict,
                                            self.model_config.spatial_dims)

        valid_transforms = build_transform_list(['filepath'],
                                                self.model_config.median_spacing,
                                                False,
                                                augm_params_dict,
                                                self.model_config.spatial_dims)
        
        return train_transforms, valid_transforms