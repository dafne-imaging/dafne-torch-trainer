import torch.nn as nn
import monai.networks.nets as monai_nets
from monai.networks.blocks import Convolution, ResidualUnit

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

        self.spatial_dims = spatial_dims
        self.out_channels = out_channels

        # define the strides for each level
        if self.spatial_dims == 3:
            strides_list = [(1, 1, 1)]
            for i in range (1, n_levels):
                if i<=2:
                    strides_list.append((2, 2, 2))
                else:
                    strides_list.append((1, 2, 2))
            strides = tuple(strides_list)
        else:
            strides = tuple([1] + [2] * (n_levels - 1))

        self.unet_model = monai_nets.Unet(
            spatial_dims = spatial_dims,
            in_channels = in_channels,
            out_channels = out_channels, 
            channels = feature_channels,
            num_res_units = num_res_units,
            strides = strides,
            kernel_size=kernel_size,
            norm='INSTANCE'
        )

    def forward(self, x):
        return self.unet_model(x)
    
    def update_output_channels(self, n_classes:int):
        m = self.unet_model
        old_up = m.model[-1]

        if isinstance(old_up, nn.Sequential):
            old_conv_block, has_ru = old_up[0], True
        else:
            old_conv_block, has_ru = old_up, False

        old_conv = old_conv_block.conv
        in_channels = old_conv.in_channels
        stride = m.strides[0]

        new_conv_block = Convolution(
            m.dimensions, in_channels, n_classes,
            strides=stride, kernel_size=m.up_kernel_size,
            act=m.act, norm=m.norm, dropout=m.dropout, bias=m.bias,
            conv_only=not has_ru, is_transposed=True,
            adn_ordering=m.adn_ordering,
        )

        if has_ru:
            new_ru = ResidualUnit(
                m.dimensions, n_classes, n_classes,
                strides=1, kernel_size=m.kernel_size, subunits=1,
                act=m.act, norm=m.norm, dropout=m.dropout, bias=m.bias,
                last_conv_only=True, adn_ordering=m.adn_ordering,
            )
            new_up = nn.Sequential(new_conv_block, new_ru)
        else:
            new_up = new_conv_block

        new_up = new_up.to(old_conv.weight.device)
        m.model[-1] = new_up
        m.out_channels = n_classes
        self.out_channels = n_classes
    

# here users can define other model classes if needed
class DafneDynUnetModel(nn.Module):
    def __init__(self,
                spatial_dims,
                in_channels:int,
                out_channels:int,
                kernels:list,
                strides:list,
                deep_supervision:bool=False,
                norm_name=("INSTANCE", {"affine": True})
                ):
        
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels

        self.dyn_unet = monai_nets.DynUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            deep_supervision=deep_supervision,
            norm_name=norm_name,
            res_block=True
        )
    
    def forward(self, x):
        return self.dyn_unet(x)
    
    def update_output_channels(self, n_classes:int):
        self.dyn_unet.out_channels = n_classes
        self.dyn_unet.output_block = \
            self.dyn_unet.get_output_block(0)

        if hasattr(self.dyn_unet, 'deep_supervision') and self.dyn_unet.deep_supervision:
            self.dyn_unet.deep_supervision_heads = \
                self.dyn_unet.get_deep_supervision_heads()
        
        self.out_channels = n_classes