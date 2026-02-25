import torch
import numpy as np


def get_optimal_hyperparameters(
    median_shape: list,
    spatial_dims: int = 3,
    safety_factor: float = 0.70,
    is_finetune: bool = False,
    mixed_precision: bool = False,
    divisor: list = None,
) -> list:
    """
    Calculates the optimal patch size and batch size constrained by available VRAM,
    taking into account U-Net architectural memory requirements.

    Args:
        median_shape (list): Median volume shape of the dataset [D, H, W] (3-D)
                             or [H, W] (2-D).
        spatial_dims (int): 2 or 3.
        safety_factor (float): Fraction of total VRAM considered usable.
                               Defaults to 0.70.
        is_finetune (bool): Whether this is a fine-tuning run (fewer active
                            gradients → lower per-voxel cost).
        mixed_precision (bool): if mixed precision is enabled.
        divisor (list): list of divisors for each dimension.

    Returns:
        tuple:
            patch_size (list[int]): Patch dimensions rounded down to the
                                    nearest multiple of 32.
            batch_size (int): Batch size in [1, 8].
    """

    if not torch.cuda.is_available():
        default_patch = [32, 128, 128] if spatial_dims == 3 else [256, 256]
        return default_patch, 2

    device = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(device)
    total_vram = gpu_props.total_memory  # bytes

    BYTES_PER_VOXEL = 6_000 if is_finetune else 10_000
    BYTES_PER_VOXEL = BYTES_PER_VOXEL * 0.6 if mixed_precision else BYTES_PER_VOXEL

    # Static overhead: model weights + optimiser state + CUDA/cuDNN buffers.
    STATIC_OVERHEAD = int(1.0e9) if is_finetune else int(1.5e9)  # 1 GB / 1.5 GB
    STATIC_OVERHEAD = STATIC_OVERHEAD * 0.8 if mixed_precision else STATIC_OVERHEAD

    usable_vram = max(total_vram * safety_factor - STATIC_OVERHEAD, 0)

    
    # Determine patch size and batch size jointly.
    # Strategy:
    #   1. Try batch_size = 2 first (good default for medical imaging).
    #   2. If the median volume fits with batch = 2, try to raise batch.
    #   3. If it does not fit, scale the patch down uniformly so that
    #      batch = 2 fits exactly within the VRAM budget.
    DESIRED_BATCH = 2
    
    if divisor is None:
        divisor = [32] * spatial_dims
    
    divisor = np.array(divisor)

    target_patch = np.array(median_shape, dtype=float)
    target_voxels = float(np.prod(target_patch))

    # Voxels that fit in VRAM for the desired batch
    max_voxels_per_sample = usable_vram / (BYTES_PER_VOXEL * DESIRED_BATCH)

    if target_voxels > max_voxels_per_sample:
        # Patch is too large → scale down uniformly while preserving aspect ratio
        scale_factor = (max_voxels_per_sample / target_voxels) ** (1.0 / spatial_dims)
        optimal_patch = target_patch * scale_factor
        optimal_batch_size = DESIRED_BATCH
    else:
        # Patch fits → keep it and try to increase batch size
        optimal_patch = target_patch
        # How many samples fit if we keep the same patch?
        potential_batch = int(usable_vram / (target_voxels * BYTES_PER_VOXEL))
        optimal_batch_size = max(1, min(potential_batch, 8))

    # Round patch dimensions down to nearest multiple of DIVISOR (required by
    # most U-Net implementations for valid pooling/upsampling paths).
    optimal_patch = np.floor(optimal_patch / divisor) * divisor
    optimal_patch = np.maximum(optimal_patch, divisor)

    # After rounding, perform a final sanity-check: does the rounded patch
    # still fit within budget at the chosen batch size?  If not, halve the
    # batch size once more (floor at 1).
    rounded_voxels = float(np.prod(optimal_patch))
    required_vram = rounded_voxels * BYTES_PER_VOXEL * optimal_batch_size + STATIC_OVERHEAD
    if required_vram > total_vram * safety_factor and optimal_batch_size > 1:
        optimal_batch_size = max(1, optimal_batch_size // 2)

    return optimal_patch.astype(int).tolist(), optimal_batch_size