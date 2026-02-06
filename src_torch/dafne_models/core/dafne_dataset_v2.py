import numpy as np
import torch 

from monai.data import Dataset, DataLoader
from monai.data.utils import list_data_collate, pad_list_data_collate

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    MapTransform,
    LoadImage,
    ToTensord,
    Resized,
    RandRotate90d,
    RandFlipd,
    RandZoomd, 
    RandGaussianNoised,
    RandCropByPosNegLabeld,
    Spacingd,
    CropForegroundd
    )

class MapTransformLoadData(MapTransform):
    '''
    Custom MONAI Transform to handle hybrid data loading strategies.
    It allow the pipeline to seamlessly handle both legacy .npz 
    archives (containing paired image/mask) and standard medical 
    formats (NIfTI, DICOM) stored in separate files.
    '''
    def __init__(self, keys, allow_missing_keys=False, spatial_dims:int=3):
        '''
        Args:
            keys (list): keys to processing in the data dictionary
            allow_missing_keys: it does not raise exception if key is missing
        '''
        super().__init__(keys, allow_missing_keys)
        self.spatial_dims = spatial_dims

    def __call__(self, data):
        '''
        Apply the transform to one sample (dictionary)
        '''
        d = dict(data)
        for key in self.keys:
            filepath = d[key]
            index = data.get('index', None)

            try: 
                with np.load(filepath) as npz_data: 
                    keys = list(npz_data.keys())
                    
                    mask_keys = sorted(k for k in npz_data.keys() if k.startswith('mask'))
                    img = npz_data['data'].astype(np.float32)
                    img = np.ascontiguousarray(np.moveaxis(img, -1, 0))
                    mask = np.zeros_like(img, dtype=np.float32)
                    current_res = npz_data['resolution']

                    for i, k in enumerate(mask_keys):
                        m = npz_data[k].astype(np.float32)
                        m = np.ascontiguousarray(np.moveaxis(m, -1, 0))
                        mask[m > 0] = i + 1

                    if self.spatial_dims == 2: 
                        img = img[index]
                        mask = mask[index]
                        current_res = np.array([current_res[0], current_res[1]], dtype=np.float32)
                    if self.spatial_dims == 3:
                        current_res = np.array([current_res[2], current_res[0], current_res[1]], dtype=np.float32)

                    d['image'] = img
                    d['mask'] = mask
                    d['image_meta_dict'] = {
                        "pixdim": np.array([1, *current_res], dtype=np.float32)

                    }
                    d['mask_meta_dict'] = {
                        "pixdim": np.array([1, *current_res], dtype=np.float32)
                    }
        
            except Exception as e: 
                print(f'Error during volume loading: {e}')

        return d


class DafneDataset(Dataset): 
    '''
        Class for loading npz dataset
        Dataset is defined a CacheDaset RAM caching.
    '''
    
    def __init__(self, 
                 data_files:list, 
                 augm_params:dict=None,
                 train_transform:bool=True,
                 spatial_dims:int=2
                 ):
        '''
        Args: 
            data_files (list): data path to anatomical images 
            (.npz files with image, masks and resolution information)
        '''
        self.data_files = data_files
        self.augm_params = augm_params if augm_params is not None else {}
        self.train_transform = train_transform
        self.spatial_dims = spatial_dims
        self.keys_to_load = ['filepath']

        data_dict = []

        if self.spatial_dims == 3: 
            data_dict = [{'filepath': f} for f in self.data_files] 
            transforms_list = self._transform_3d_data()

        elif self.spatial_dims == 2: 
            for f in self.data_files:
                with np.load(f) as npz_data:
                    depth = npz_data['data'].shape[2]
                    for d in range(depth):
                        data_dict.append({'filepath':f, 'index': d})
            transforms_list = self._transform_2d_data()
            
        self.transform = Compose(transforms_list)

        super().__init__(data=data_dict, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def _transform_3d_data(self):
        pipeline = [
            MapTransformLoadData(keys=self.keys_to_load, spatial_dims=3),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            ScaleIntensityd(keys=['image']),
        ]

        if self.train_transform:
            pipeline.append( 
                RandCropByPosNegLabeld(
                    keys=['image', 'mask'], label_key='mask',
                    spatial_size=(16, 96, 96), # Patch 3D
                    pos=1, neg=1, num_samples=4,
                    image_key='image', image_threshold=0
                )
            )
            
            if self.augm_params.get('rotate'):
                pipeline.append(RandRotate90d(keys=['image', 'mask'], prob=0.5, spatial_axes=(1,2)))
            if self.augm_params.get('flip_x'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1))
            if self.augm_params.get('flip_y'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2))
            if self.augm_params.get('zoom'):
                pipeline.append(RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=['bilinear', 'nearest']))
            if self.augm_params.get('noise'):
                pipeline.append(RandGaussianNoised(keys=['image'], prob=0.5, std=0.05))

        pipeline.append(ToTensord(keys=['image', 'mask']))
        return pipeline
    
    def _transform_2d_data(self):
        pipeline = [
            MapTransformLoadData(keys=self.keys_to_load, spatial_dims=2),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0), mode=('bilinear', 'nearest')),
            ScaleIntensityd(keys=['image']),
            #Resized(keys=['image', 'mask'], spatial_size=(256, 256), mode=['bilinear', 'nearest'])
        ]
        
        if self.train_transform:
            if self.augm_params.get('rotate'):
                pipeline.append(RandRotate90d(keys=['image', 'mask'], prob=0.5, spatial_axes=(0,1)))
            if self.augm_params.get('flip_x'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0))
            if self.augm_params.get('flip_y'):
                pipeline.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1))
            if self.augm_params.get('zoom'):
                pipeline.append(RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=['bilinear', 'nearest']))
            if self.augm_params.get('noise'):
                pipeline.append(RandGaussianNoised(keys=['image'], prob=0.5, std=0.05))
        
        pipeline.append(ToTensord(keys=['image', 'mask']))
        return pipeline


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt

    # test cachedataset
    root_data_dir = "/Users/giuseppetimpano/Desktop/Project code/dafne-project/Test_images/Data_Giuseppe" 
    
    all_npz_files = []
    for root, dirs, files in os.walk(root_data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_npz_files.append(os.path.join(root, file))
    
    all_npz_files.sort()

    try:
        dataset = DafneDataset(data_files=all_npz_files, spatial_dims=3, train_transform=True)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=pad_list_data_collate)
        print("Dataset correctly done")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        print("\n--- First data ---")
        '''count_label_2d = set()
        for batch in dataset: 
            count_label_2d.update(np.unique(batch['mask']).astype(np.int8))
        print(count_label_2d)'''
        '''first_sample = list(dataset)[6]
        img = first_sample['image']
        mask = first_sample['mask']
        print(img.shape, mask.shape, np.unique(mask))

        print(f"File: {all_npz_files[0]}")
        print(f"Shape Immagine: {img.shape}") # (C, H, W)
        print(f"Shape Maschera: {mask.shape}")
        print(f"Tipo Dati: {img.dtype}")
        
        if len(img.shape) == 3:
            plt.imshow(img[0, :, :], cmap='gray')
            plt.imshow(mask[0, :, :], cmap='gray', alpha=0.5)
            plt.show()
        else: 
            plt.imshow(img[0, 7, :, :], cmap='gray')
            plt.imshow(mask[0, 7, :, :], cmap='gray', alpha=0.5)
            plt.show()'''

        for batch_idx, batch in enumerate(loader):
            images = batch['image']
            masks = batch['mask']
            print(f"Batch {batch_idx}: images {images.shape}, masks {masks.shape}, pixdim {batch['image_meta_dict']['pixdim']}, n_classes {torch.unique(batch['n_classes'])}")
            break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()