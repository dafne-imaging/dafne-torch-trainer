import torch
import numpy as np

def get_optimal_hyperparameters(median_shape: list, spatial_dims: int = 3, safety_factor: float = 0.80) -> list:
    """
    Calculates the optimal patch size constrained by VRAM and U-Net architectural requirements.
        
    Args:
        median_shape (list): The median shape of the dataset [D, H, W].
        spatial_dims (int): 2 or 3 dimensions.
        safety_factor (float): Fraction of VRAM to use (default 0.80).
        
    Returns:
        list: The calculated patch size [D, H, W] (integers) and best batch size
    """

    if not torch.cuda.is_available():
        # Fallback CPU
        default_patch = [32, 128, 128] if spatial_dims == 3 else [256, 256]
        return default_patch, 2
    
    device = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(device)
    total_vram = gpu_props.total_memory
    
    BYTES_PER_VOXEL = 5500 
    STATIC_OVERHEAD = 500 * 1024 * 1024 
    
    usable_vram = (total_vram * safety_factor) - STATIC_OVERHEAD
    optimal_batch_size = 2
    max_voxels_budget = usable_vram / (BYTES_PER_VOXEL * optimal_batch_size)
    
    target_patch = np.array(median_shape, dtype=float)
    target_voxels = np.prod(target_patch)
    
    if target_voxels > max_voxels_budget:
        root = 1/spatial_dims
        scale_factor = (max_voxels_budget / target_voxels) ** root
        optimal_patch = target_patch * scale_factor
    else:
        optimal_patch = target_patch
        potential_batch = usable_vram / (target_voxels * BYTES_PER_VOXEL)
        optimal_batch_size = int(potential_batch)
        optimal_batch_size = min(optimal_batch_size, 8)
        optimal_batch_size = max(optimal_batch_size, 2)

    DIVISOR = 32
    optimal_patch = np.floor(optimal_patch / DIVISOR) * DIVISOR
    optimal_patch = np.maximum(optimal_patch, [DIVISOR] * spatial_dims)
    
    return optimal_patch.astype(int).tolist(), optimal_batch_size