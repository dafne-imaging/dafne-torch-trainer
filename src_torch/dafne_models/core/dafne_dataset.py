import numpy as np
import torch 

from monai.data import Dataset, DataLoader
from monai.data.utils import list_data_collate, pad_list_data_collate

from monai.transforms import (
    Compose,
    MapTransform
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
                    mask = np.zeros_like(img, dtype=np.uint8)
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
                 spatial_dims:int=2,
                 dyn_unet:bool=False,
                 external_transforms=None
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
        self.external_transforms = external_transforms

        data_dict = []

        if self.spatial_dims == 3: 
            data_dict = [{'filepath': f} for f in self.data_files] 

        elif self.spatial_dims == 2: 
            for f in self.data_files:
                with np.load(f) as npz_data:
                    depth = npz_data['data'].shape[2]
                    for d in range(depth):
                        data_dict.append({'filepath':f, 'index': d})
            
        if self.external_transforms is not None:
            self.transform = self.external_transforms
        else:
            self.transform = None
            raise ValueError('Any kind of transforms are defined for data!')

        super().__init__(data=data_dict, transform=self.transform)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt

    # test dataset and dataloader
    root_data_dir = "" 
    
    all_npz_files = []
    for root, dirs, files in os.walk(root_data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_npz_files.append(os.path.join(root, file))
    
    all_npz_files.sort()

    try:
        dataset = DafneDataset(data_files=all_npz_files, spatial_dims=3, train_transform=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=pad_list_data_collate)
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
        img = first_sample[0]['image']
        mask = first_sample[0]['mask']
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
            plt.show()
'''
        for batch_idx, batch in enumerate(loader):
            images = batch['image']
            masks = batch['mask']
            print(f"Batch {batch_idx}: images {images.shape}, masks {masks.shape}, pixdim {batch['image_meta_dict']['pixdim']}")
            break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()