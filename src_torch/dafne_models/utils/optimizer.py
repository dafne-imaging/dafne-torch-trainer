import torch
import numpy as np

def get_optimal_patch_size(median_shape: list, spatial_dims: int = 3, safety_factor: float = 0.80) -> list:
    """
    Calculates the optimal patch size constrained by VRAM and U-Net architectural requirements.
        
    Args:
        median_shape (list): The median shape of the dataset [D, H, W].
        spatial_dims (int): 2 or 3 dimensions.
        safety_factor (float): Fraction of VRAM to use (default 0.80).
        
    Returns:
        list: The calculated patch size [D, H, W] (integers).
    """

    if not torch.cuda.is_available():
        return [32, 128, 128] if spatial_dims == 3 else [256, 256]
    
    device = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(device)
    total_vram = gpu_props.total_memory
    
    BYTES_PER_VOXEL = 5500 
    BATCH_SIZE = 2
    STATIC_OVERHEAD = 500 * 1024 * 1024 
    
    usable_vram = (total_vram * safety_factor) - STATIC_OVERHEAD
    
    max_voxels_budget = usable_vram / (BYTES_PER_VOXEL * BATCH_SIZE)
    
    current_patch = np.array(median_shape, dtype=float)
    current_voxels = np.prod(current_patch)
    
    if current_voxels > max_voxels_budget:
        root = 1/spatial_dims
        scale_factor = (max_voxels_budget / current_voxels) ** root
        current_patch = current_patch * scale_factor

    DIVISOR = 32
    optimal_patch = np.floor(current_patch / DIVISOR) * DIVISOR
    
    optimal_patch = np.maximum(optimal_patch, [DIVISOR] * spatial_dims)
    
    return optimal_patch.astype(int).tolist()