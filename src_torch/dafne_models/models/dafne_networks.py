import torch
import torch.nn as nn
import monai.networks.nets as monai_nets

# just defined a model class: Unet from MONAI framework

class DafneUnetModel(nn.Module):
    # full Unet model from MONAI pytorch framwwork with parametrizable depth, kernel size, in_channels and out_channels
    def __init__(self, spatial_dims,
                    in_channels:int,
                    out_channels:int,
                    start_channel:int=32,
                    n_levels:int=5, 
                    num_res_units:int=2,
                    kernel_size:int=3):
        super().__init__()

        # define the feature channels extracted on downsampling path: it depends on n_levels
        feature_channels = tuple(start_channel * (2**i) for i in range(n_levels))

        # define the strides for each level
        # first level no downsampling, then downsample by factor 2
        strides = tuple([1] + [2] * (n_levels - 1))

        self.unet_model = monai_nets.Unet(
            spatial_dims = spatial_dims,
            in_channels = in_channels,
            out_channels = out_channels, 
            channels = feature_channels,
            num_res_units = num_res_units,
            strides = strides,
            kernel_size=kernel_size,
            norm='batch'
        )

    def forward(self, x):
        return self.unet_model(x)
    
# here users can define other model classes if needed


class DafneDynUnet(nn.Module):
    def __init__(self,
                in_channels:int,
                out_channels:int,
                kernel_size:list,
                strides:list,
                ):
        
        super().__init__()

        self.dyn_unet = monai_nets.DynUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=strides[:1],
            deep_supervision=True,
            deep_supr_num=1, 
            res_block=True
        )
    
    def forward(self, x):
        return self.dyn_unet(x)