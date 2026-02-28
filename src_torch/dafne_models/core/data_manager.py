import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from monai.data.utils import pad_list_data_collate
from monai.data import DataLoader, Dataset

from ..config.config_params import (DatasetConfig, 
                                    TrainingConfig, 
                                    ModelConfig,
                                    AugmentationConfig)


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
                with np.load(f, mmap_mode='r') as npz_data:
                    depth = npz_data['data'].shape[2]
                    has_mask = np.zeros(shape=depth, dtype=bool)
                    mask_keys = sorted(k for k in npz_data.keys() if k.startswith('mask'))
                    for mask_key in mask_keys:
                        has_mask |= np.any(npz_data[mask_key], axis=(0,1))
                    indices = set()
                    positive_indices = np.where(has_mask)[0]
                    for i in positive_indices:
                        indices.update([i-1, i, i+1])
                    indices = sorted(i for i in indices if i>=0 and i<depth)

                    for i in indices:
                        data_dict.append({
                            'filepath': f,
                            'index': i
                        })
            
        if self.external_transforms is not None:
            self.transform = self.external_transforms
        else:
            self.transform = None
            raise ValueError('Any kind of transforms are defined for data!')

        super().__init__(data=data_dict, transform=self.transform)
                   
    def __len__(self):
        return len(self.data)


class DafneDataModule():
    def __init__(self, 
                dataset_config: DatasetConfig,
                model_config: ModelConfig,
                training_config: TrainingConfig
                ):
        
        # config classes for data and model
        self.dataset_config = dataset_config # dataset config class
        self.training_config = training_config
        self.model_config = model_config
        self.augmentation_config = training_config.augmentation
        
        self.train_files = [] # train files list
        self.val_files = [] # val files list
        self.data_list = [] # get all the files from root_dir

        self.train_dataset = None
        self.val_dataset = None

        self.train_loader = None
        self.val_loader = None


    def _scan_directory_folds(self, extensions):
        '''
        Scan the directory for files with the given extensions
        '''
        root_dir = self.dataset_config.root_dir
        found_files = []
        try: 
            for root, _, files in os.walk(root_dir, topdown=True):
                for file in files: 
                    if any(file.endswith(ext) for ext in extensions):
                        full_path = os.path.join(root, file)
                        found_files.append(full_path)
            found_files.sort()
            return found_files
        except Exception as e:
            print(f"Error scanning directory {root_dir}: {e}")
            return []
    
    def get_original_data(self):
        return self._scan_directory_folds(extensions=['.npz'])
    
    def setup(self, extensions:list = None):
        '''
        Setup the dataset
        '''
        exts = extensions or ['.npz']
        self.data_list = self._scan_directory_folds(exts)
        self.train_files, self.val_files = train_test_split(self.data_list, 
                                test_size=self.dataset_config.val_split, 
                                random_state=self.dataset_config.random_seed)

    def create_datasets(self, train_transforms, val_transforms):
        '''
        Create datasets
        '''
        train_dataset = DafneDataset(data_files=self.train_files, 
                                    spatial_dims=self.model_config.spatial_dims,
                                    train_transform=True,
                                    external_transforms=train_transforms)
        val_dataset = DafneDataset(data_files=self.val_files, 
                                    spatial_dims=self.model_config.spatial_dims,
                                    train_transform=False,
                                    external_transforms=val_transforms)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    def create_dataloaders(self):
        '''
        Create dataloaders
        '''
        self.train_loader = DataLoader(self.train_dataset, 
                                    batch_size=self.training_config.batch_size, 
                                    shuffle=True,
                                    collate_fn=pad_list_data_collate)
        
        self.val_loader = DataLoader(self.val_dataset, 
                                    batch_size=1, 
                                    shuffle=False,
                                    collate_fn=pad_list_data_collate)

    def get_dataloaders(self):
        '''
        Get dataloaders
        '''
        return self.train_loader, self.val_loader

