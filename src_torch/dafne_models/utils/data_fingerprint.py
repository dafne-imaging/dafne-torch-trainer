import logging
import numpy as np

logger = logging.getLogger(__name__)


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
                logger.warning("Error reading %s: %s", filepath, e)

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
                logger.warning("Error reading %s: %s", filepath, e)

        if not shapes:
            raise ValueError("Could not extract any shape from the dataset list.")

        return np.median(np.array(shapes), axis=0).astype(int)

    def get_kernel_and_strides(self, patch_size:list=None):

        '''
        Return kernels and strides list based on median data spacing and data shape
        
        :param patch_size: median dataset spacing
        :param data_shape: median data shape
        '''

        if patch_size is not None:
            sizes = np.array(patch_size, dtype=float)
        else:
            sizes = np.array(self.data_shape, dtype=float)
            
        spacings = np.array(self.data_spacing)

        strides, kernels = [], []
        

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
    
    def get_labels_name(self) -> list[str]:
        labels_name = set()

        for i in range(len(self.data_list)):
            filepath = self.data_list[i]
            try: 
                with np.load(filepath, mmap_mode='r') as npz_data:
                    keys = list(npz_data.keys())
                    labels_name.update(k for k in keys if k.startswith('mask'))
            except Exception as e:
                logger.warning("Could not read labels from %s: %s", filepath, e)

        return sorted(list(labels_name))

    def compute_class_weights(self, n_classes: int) -> np.ndarray:
        '''
        Compute inverse-frequency class weights from the training masks.
        Returns a normalized weight array of shape [n_classes] (class 0 = background).
        '''
        voxel_counts = np.zeros(n_classes, dtype=np.float64)

        for filepath in self.data_list:
            try:
                with np.load(filepath, mmap_mode='r') as npz_data:
                    mask_keys = sorted([k for k in npz_data.keys() if k.startswith('mask')])
                    if not mask_keys:
                        continue
                    total_voxels = npz_data[mask_keys[0]].size
                    fg_counts = np.array([npz_data[k].sum() for k in mask_keys[:n_classes - 1]], dtype=np.float64)
                    voxel_counts[0] += total_voxels - fg_counts.sum()
                    voxel_counts[1:1 + len(fg_counts)] += fg_counts
            except Exception as e:
                logger.warning("Skipping %s for class weights: %s", filepath, e)

        total = voxel_counts.sum()
        if total == 0:
            return np.ones(n_classes, dtype=np.float32) / n_classes

        freq = voxel_counts / total
        weights = 1.0 / np.sqrt(freq + 1e-6)
        weights = weights / weights.sum()
        return weights.astype(np.float32)

    @staticmethod
    def count_label_mask(data_list: list[str]) -> int:
        max_masks_found = 0
    
        for i in range(len(data_list)):
            filepath = data_list[i]
            try:
                # mmap_mode='r' legge solo l'header del file
                with np.load(filepath, mmap_mode='r') as npz:
                    keys = list(npz.keys())
                    # Conta quante chiavi iniziano con 'mask'
                    n_masks = len([k for k in keys if k.startswith('mask')])
                    
                    if n_masks > max_masks_found:
                        max_masks_found = n_masks
            except Exception as e:
                logger.warning("Could not count masks in %s: %s", filepath, e)

        total_classes = max(2, max_masks_found + 1)

        return total_classes