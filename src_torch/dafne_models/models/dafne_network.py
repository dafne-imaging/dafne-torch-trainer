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

if __name__ == "__main__":
    try:        
        # Test Full UNet
        model_full = DafneUnetModel(spatial_dims=2, in_channels=1, out_channels=2, n_levels=5)
        print("DafneUnetModel correctly instantiated.")
        count = 0
        # Calcoliamo la profondità massima analizzando tutti i nomi
        for name, layer in model_full.named_modules():
            if isinstance(layer, nn.Conv2d):
                print(layer, name)

        #print(model_full)
        # Test Dummy Input
        #x = torch.randn(1, 1, 128, 128) # Batch=1, Canale=1, H=128, W=128
        #y = model_full(x)
        #print(f"Forward pass OK. Output shape: {y.shape}")
        
    except Exception as e:
        print(f"Critical error: {e}")