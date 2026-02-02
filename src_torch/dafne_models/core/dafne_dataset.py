import numpy as np

from monai.data import CacheDataset
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
    RandGaussianNoised
    )

class MapTransformLoadData(MapTransform):
    '''
    Custom MONAI Transform to handle hybrid data loading strategies.
    It allow the pipeline to seamlessly handle both legacy .npz 
    archives (containing paired image/mask) and standard medical 
    formats (NIfTI, DICOM) stored in separate files.
    '''
    def __init__(self, keys, allow_missing_keys=True):
        '''
        Args:
            keys (list): keys to processing in the data dictionary
            allow_missing_keys: it does not raise exception if key is missing
        '''
        super().__init__(keys, allow_missing_keys)
        self.monai_loader = LoadImage(image_only=True, ensure_channel_first=False)

    def __call__(self, data):
        '''
        Apply the transform to one sample (dictionary)
        '''
        d = dict(data)
        for key in self.keys:
            filepath = d[key]

            try: 
                if filepath.endswith('.npz'):
                    try:
                        with np.load(filepath) as npz_data:
                            d['image'] = npz_data['arr_0'].astype(np.float32)
                            d['mask'] = npz_data['arr_1'].astype(np.float32)
                    except Exception as e:
                        print(f'Error occured during .npz reading')

                elif filepath.endswith(('.nii', '.dcm', '.nii.gz')):
                    try:
                        d[key] = self.monai_loader(filepath).astype(np.float32)
                    except Exception as e: 
                        print(f'Error occurred during loading {filepath}: {e}')
            except Exception as e:
                print(f'Image format not supported: {e}')
        return d


class DafneCacheDataset(CacheDataset): 
    '''
        Class for loading npz dataset
        Dataset is defined a CacheDaset RAM caching.
    '''
    
    def __init__(self, 
                 image_files:list, 
                 cache_rate=1.0, 
                 mask_files:list=None,
                 augm_params:dict=None,
                 train_transform:bool=True):
        '''
        Args: 
            image_files (list): data path to anatomical images (.nii or .npz images)
            mask_files (list)=None: data path to masks (optional for .npz data) 
        '''
        self.image_files = image_files
        self.mask_files = mask_files
        self.augm_params = augm_params if augm_params is not None else {}
        self.train_transform = train_transform

        if self.mask_files is not None and len(self.mask_files) > 0:
            if len(self.image_files) != len(self.mask_files):
                raise ValueError("Number of images and masks do not correpond")
            data_dict = [{"image":image, "mask": mask} 
                          for image, mask in zip(self.image_files, self.mask_files)
                          ]
            keys_to_load = ['image', 'mask']
        elif self.mask_files is None: 
            # create a dict of path
            data_dict = [{'file_path':path} for path in image_files]
            keys_to_load = ['file_path']

        base_transforms = [
            MapTransformLoadData(keys=keys_to_load),
            EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
            ScaleIntensityd(keys=['image']),
            ToTensord(keys=['image', 'mask']),
            Resized(
                keys=['image', 'mask'], 
                spatial_size=(256, 256), 
                mode=['bilinear', 'nearest'] 
            )
        ]

        add_transforms = []

        if self.augm_params.get('rotate'):
            add_transforms.append(RandRotate90d(keys=['image', 'mask'], prob=0.5, spatial_axes=0))
        if self.augm_params.get('flip_x'):
            add_transforms.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0))
        if self.augm_params.get('flip_y'):
            add_transforms.append(RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1))
        if self.augm_params.get('zoom'):
            add_transforms.append(RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=['bilinear', 'nearest']))
        if self.augm_params.get('noise'):
            add_transforms.append(RandGaussianNoised(keys=['image'], prob=0.5, std=0.05))

        if self.train_transform and self.augm_params is not None:
            transform = base_transforms + add_transforms
            self.transform = Compose(transform)
        elif not self.train_transform or self.augm_params is None:
            self.transform = Compose(base_transforms)

        super().__init__(data=data_dict, transform=self.transform, cache_rate=cache_rate)

    def __len__(self):
        return len(self.image_files)
    
if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt

    # test cachedataset
    root_data_dir = "/Users/giuseppetimpano/Desktop/Project code/dafne-project/Test_images/npz_test" 
    
    all_npz_files = []
    for root, dirs, files in os.walk(root_data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_npz_files.append(os.path.join(root, file))
    
    all_npz_files.sort()

    try:
        dataset = DafneCacheDataset(image_files=all_npz_files, mask_files=None, cache_rate=0.0)
        print("Dataset correctly done")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        print("\n--- ANALISI PRIMO CAMPIONE ---")
        first_sample = dataset[76]
        img = first_sample['image']
        mask = first_sample['mask']
        print(np.unique(mask))

        print(f"File: {all_npz_files[0]}")
        print(f"Shape Immagine: {img.shape}") # (C, H, W)
        print(f"Shape Maschera: {mask.shape}")
        print(f"Tipo Dati: {img.dtype}")
        
        # Opzionale: Visualizzazione rapida se sei in un ambiente grafico
        plt.imshow(img[0, :, :], cmap='gray')
        plt.imshow(mask[0, :, :], cmap='gray', alpha=0.5)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()