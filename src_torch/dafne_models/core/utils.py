import numpy as np


def get_median_spacing(data_list:dict, spatial_dims:int=2):
    
    pixel_spacing_list = []
    for data in data_list:
        try:
            with (np.load(data, mmap_mode='r')) as npz_data:
                curr_res = npz_data['resolution'] 
                if spatial_dims == 3: 
                    pixel_spacing_list.append([curr_res[2], curr_res[0], curr_res[1]])
                elif spatial_dims == 2: 
                    pixel_spacing_list.append(curr_res[:2])
        except Exception as e:
            print(f'error during spacing extraction: {e}')
    
    return np.median(pixel_spacing_list, axis=0).tolist()


def count_label_mask(data_list: list):
    max_masks_found = 0
    limit = min(len(data_list), 50)
    
    for i in range(limit):
        filepath = data_list[i]
        try:
            # mmap_mode='r' legge solo l'header del file
            with np.load(filepath, mmap_mode='r') as npz:
                keys = list(npz.keys())
                # Conta quante chiavi iniziano con 'mask'
                n_masks = len([k for k in keys if k.startswith('mask')])
                
                if n_masks > max_masks_found:
                    max_masks_found = n_masks
        except Exception:
            continue
    
    total_classes = max_masks_found + 1
    return max(2, total_classes)