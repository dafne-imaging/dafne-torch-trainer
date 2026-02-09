import os
import sys

import numpy as np
import torch

import torch
import numpy as np


class DatasetFingerprint():

    def __init__(self, data_list:list, spatial_dims:int=3):

        self.data_list = data_list
        self.spatial_dims = spatial_dims

        self.data_spacing = self._get_median_spacing()
        self.data_shape = self._get_median_shape()
    
    def _get_median_spacing(self):
        spacings = []

        for filepath in self.data_list:

            try:
                with np.load(filepath, mmap_mode='r') as npz_data:
                    res = npz_data['resolution'] #resolution key in .npz data
            
                    if self.spatial_dims == 3: 
                        spacings.append([res[2], res[0], res[1]])
                    elif self.spatial_dims == 2: 
                        spacings.append(res[:2])
            
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")
        
        if not spacings:
            raise ValueError("Could not extract any spacing from the dataset list.")

        return np.median(np.array(spacings), axis=0).astype(np.float16)

    def _get_median_shape(self):
        shapes = []

        for filepath in self.data_list:

            try:
                with np.load(filepath, mmap_mode='r') as npz_data:
                    shape = npz_data['data'].shape #resolution key in .npz data
            
                    if self.spatial_dims == 3: 
                        shapes.append([shape[2], shape[0], shape[1]])
                    elif self.spatial_dims == 2: 
                        shapes.append(shape[:2])
            
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")
        
        if not shapes:
            raise ValueError("Could not extract any spacing from the dataset list.")

        return np.median(np.array(shapes), axis=0).astype(int)

    def get_kernel_and_strides(self, patch_size:list=None):

        '''
        Return kernels and strides list based on median data spacing and data shape
        
        :param data_spacing: median dataset spacing
        :param data_shape: median data shape
        '''

        if patch_size is not None:
            sizes = np.array(patch_size, dtype=float)
        else:
            sizes = np.array(self.data_shape, dtype=float)
        
        strides, kernels = [], []
        sizes = np.array(self.data_shape)
        spacings = np.array(self.data_spacing)

        while(True):
            spacing_ratio = [sp / min(spacings) for sp in spacings]

            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all (s==1 for s in stride):
                break
            
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
        
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])

        return kernels, strides