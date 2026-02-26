# define utils transform to manage the pixel spacing
# inspired by https://github.com/Project-MONAI/tutorials/blob/main/modules/dynunet_pipeline/transforms.py#L111

import torch

import numpy as np
from skimage.transform import resize
from monai.transforms.utils import generate_spatial_bounding_box

from monai.transforms import (MapTransform, 
                            CropForegroundd, 
                            NormalizeIntensity, 
                            SpatialCrop,
                            DivisiblePadd)


def resample_image(image, shape, anisotrophy_flag):
    '''
    Docstring per resample_image
    
    :param image: image to resample
    :param shape: image shape
    :param anisotrophy_flag: True if image is anisotrophy
    '''
    if image.ndim == 4:
        image_list = image
        is_multichannel = True
    else:
        image_list = [image]
        is_multichannel = False

    resized_channels = []

    for image_c in image_list:
        # image_c: np.array(Depth, Height, Width)
        
        if anisotrophy_flag:
            resized_slices = []
            target_2d_shape = shape[1:]

            for i in range(image_c.shape[0]):
                image_c_2d_slice = image_c[i, :, :]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    target_2d_shape,
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=0)
            
            if resized.shape[0] != shape[0]:
                resized = resize(
                    resized,
                    shape,
                    order=1,
                    mode="constant",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
            resized_channels.append(resized)

        else:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    
    if is_multichannel:
        return np.stack(resized_channels, axis=0)
    else:
        return resized_channels[0]

def resample_label(label, shape, anisotrophy_flag):
    '''
    Docstring per resample_image
    
    :param image: image to resample
    :param shape: image shape
    :param anisotrophy_flag: True if image is anisotrophy
    '''
    if label.ndim == 4:
        label_vol = label[0] 
    else:
        label_vol = label

    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label_vol)
    
    if anisotrophy_flag:
        shape_2d = shape[1:]
        depth = label_vol.shape[0]
        
        reshaped_2d = np.zeros((depth, *shape_2d), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label_vol[depth_, :, :] == class_
                
                if not np.any(mask): continue

                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=0,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[depth_, :, :][resized_2d >= 0.5] = class_
        
        target_depth = shape[0]
        if depth != target_depth:
            for class_ in range(1, int(n_class) + 1):
                mask = reshaped_2d == class_
                if not np.any(mask): continue
                
                resized = resize(
                    mask.astype(float),
                    shape,
                    order=0,
                    mode="constant",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[resized >= 0.5] = class_
        else:
            reshaped = reshaped_2d

    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label_vol == class_
            if not np.any(mask): continue
            
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


class PreprocessAnisotropy(MapTransform):
    def __init__(self, 
                 keys, 
                 target_spacing,
                 clip_values=None, 
                 normalize_values=None, 
                 model_mode="train",
                 spatial_dims:int=3):
        super().__init__(keys)
        
        self.target_spacing = target_spacing
        self.spatial_dims = spatial_dims
        self.keys = keys

        self.low = clip_values[0] if clip_values else 0
        self.high = clip_values[1] if clip_values else 0
        self.mean = normalize_values[0] if normalize_values else 0
        self.std = normalize_values[1] if normalize_values else 1

        self.training = (model_mode == 'train')

        self.crop_foreg = CropForegroundd(keys=['image', 'mask'], source_key='image', allow_smaller=True)
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)

    def calculate_new_shape(self, spacing, shape):
        aspect_ratio = np.array(spacing) / self.target_spacing
        new_shape = np.round(aspect_ratio * np.array(shape)).astype(int)
        return new_shape
    
    def check_anisotrophy(self, spacing):
        def check(s):
            return np.max(spacing) / np.min(spacing) >= 3
        return check(spacing) or check(self.target_spacing)
    
    def __call__(self, data):
        d = dict(data)
        image = d['image']
        
        has_mask = "mask" in self.keys and "mask" in d
        label = d['mask'] if has_mask else None

        current_spacing = d['image_meta_dict']['pixdim']
        current_spacing = current_spacing[1:] # no channel spacing
        image_spacing = np.array(current_spacing).tolist()

        if self.training:
            cropped_data = self.crop_foreg({"image": image, "mask": label})
            if 0 in cropped_data["image"].shape[1:]:
                pass
            else:
                image, label = cropped_data["image"], cropped_data["mask"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            box_start, box_end = generate_spatial_bounding_box(image, allow_smaller=True) #return pixels coordinates where the foreground is located
            temp_image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image) #return cropped image, 0 if the box_start=box_end
            if 0 in temp_image.shape[1:]:
                d["bbox"] = np.vstack([box_start, box_end])
                d["crop_shape"] = np.array(image.shape[1:])
            else: 
                image = temp_image
                if has_mask:
                    label = SpatialCrop(roi_start=box_start, roi_end=box_end)(label)
                d["bbox"] = np.vstack([box_start, box_end])
                d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]

        resample_flag = False
        anisotrophy_flag = False

        if isinstance(image, torch.Tensor): image = image.numpy()
        if "mask" in self.keys and isinstance(label, torch.Tensor): label = label.numpy()

        if 0 not in original_shape and not np.allclose(self.target_spacing, image_spacing, atol=1e-4):
            resample_flag = True
            anisotrophy_flag = self.check_anisotrophy(image_spacing)
            
            resample_shape = self.calculate_new_shape(image_spacing, original_shape)
            
            if self.spatial_dims == 3:
                image = resample_image(image, resample_shape, anisotrophy_flag)
                if has_mask:
                    label = resample_label(label, resample_shape, anisotrophy_flag)
            elif self.spatial_dims == 2: 
                resized_channels = []
                for img_c in image:
                    res = resize(
                        img_c,
                        resample_shape,
                        order=3,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_channels.append(res)
                image = np.stack(resized_channels, axis=0)
                if has_mask:
                    resized_channel_mask = []
                    for mask_c in label:
                        mask_res = resize(
                            mask_c,
                            resample_shape,
                            order=0,
                            mode="edge",
                            cval=0,
                            clip=True,
                            anti_aliasing=False
                        )
                        resized_channel_mask.append(mask_res.astype(np.uint8))
                    label = np.stack(resized_channel_mask, axis=0)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag

        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image
        if "mask" in self.keys:
            d["mask"] = label

        return d

# delete Spacingd and ScaleIntensityd from transformations