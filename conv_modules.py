import torch
import torch.nn as nn
import torch.nn.functional as F
class SEModule(nn.Module):
    """
    Spatial squeeze & channel excitation attention module, as proposed in https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x)

class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) module.
    """
    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        cse_out = x*self.cSE(x)
        # Apply spatial attention
        sse_out = x*self.sSE(x)
        # Combine the outputs
        return torch.max(cse_out,sse_out)


class SCSEModule1d(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) module.
    """
    def __init__(self, in_channels, reduction=16):
        super(SCSEModule1d, self).__init__()
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        cse_out = x*self.cSE(x)
        # Apply spatial attention
        sse_out = x*self.sSE(x)
        # Combine the outputs
        return torch.max(cse_out,sse_out)



class SpatialTransformer(nn.Module):
    def __init__(self, in_channels,initial_transform_strength = 0.1):
        """
        Initializes the SpatialTransformer module.

        Args:
            in_channels (int): The number of input channels in the input feature map.
                This is used to configure the convolutional layers in the localization networks.
            
            initial_transform_strength (float, optional): A scaling factor for initializing the weights
                of the affine transformation parameters. This determines how much the initial transformations
                deviate from the identity transformation. Default is 0.1.

        Attributes:
            localizations (nn.ModuleList): A list of submodules that implement the localization networks 
                for each scale in the spatial pyramid. Each localization network processes the input 
                feature map at a different scale to extract spatial information.
            
            fc_loc (nn.Sequential): A fully connected network that takes the concatenated outputs from
                the spatial pyramid and predicts the affine transformation parameters. It consists of:
                - A linear layer that maps the concatenated features to 32 intermediate dimensions.
                - A ReLU activation.
                - A final linear layer that outputs 6 parameters for the affine transformation.

        Notes:
            - The affine transformation parameters are initialized to produce an identity transformation
            with slight random perturbations controlled by `initial_transform_strength`.
            - The spatial pyramid consists of downsampling scales `[1, 2, 4, 8]` to capture multi-scale
            spatial features, making the model robust to transformations at different resolutions.

        Example:
            >>> stn = SpatialTransformer(in_channels=3, initial_transform_strength=0.05)
            >>> input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
            >>> output = stn(input_tensor)
        """
        super(SpatialTransformer, self).__init__()
        # Localization network
        
        self.localizations = nn.ModuleList()
        spacial_pyramid_scales = [1,4,8,16,32]
        for scale in spacial_pyramid_scales:
            localization = nn.Sequential(
                nn.MaxPool2d(scale,padding=scale//2),
                nn.Conv2d(in_channels, 8, kernel_size=5,stride=2,padding=2),
                nn.ReLU(True),
                nn.Conv2d(8, 16, kernel_size=5,stride=2,padding=2),
                nn.ReLU(True),
                nn.AdaptiveMaxPool2d(1)
            )
            self.localizations.append(localization)
            
        
        # Fully connected layers to output affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(16*len(spacial_pyramid_scales),32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self._initialize_weights(initial_transform_strength)

    def _initialize_weights(self,initial_transform_strength):
        # Initialize the weights of the last fully connected layer to zero
        nn.init.normal_(self.fc_loc[-1].weight)
        with torch.no_grad():
            self.fc_loc[-1].weight*=initial_transform_strength
        # Initialize the bias to produce the identity affine transformation
        identity_bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        self.fc_loc[-1].bias.data.copy_(identity_bias)

    def forward(self, x):
        xs = [torch.jit.fork(l,x) for l in self.localizations]
        xs = [torch.jit.wait(l) for l in xs]
        xs = torch.stack(xs,dim=1).flatten(1) # Flatten while preserving batch size
        theta = self.fc_loc(xs) #[batch,6]
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class BSConvU(torch.nn.Sequential): 
    """
    Blueprint Pointwise-Depthwise Convolution Block.

    This block implements a convolutional module that performs pointwise convolution
    followed by depthwise convolution. It is an alternative to the standard depthwise-pointwise 
    approach and may be more effective in certain use cases.

    Attributes:
        pw (torch.nn.Conv2d): The pointwise convolution (1x1) layer.
        bn (torch.nn.BatchNorm2d, optional): Optional BatchNorm applied after the pointwise convolution.
        dw (torch.nn.Conv2d): The depthwise convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the depthwise convolution kernel.
        stride (int, optional): Stride for the depthwise convolution. Default is 1.
        padding (int or tuple, optional): Implicit padding for the depthwise convolution. Default is 0.
        dilation (int, optional): Dilation rate for the depthwise convolution. Default is 1.
        bias (bool, optional): Whether to add a learnable bias to the depthwise convolution. Default is True.
        padding_mode (str, optional): Padding mode for the depthwise convolution. Default is 'zeros'.
        with_bn (bool, optional): If True, includes BatchNorm between pointwise and depthwise convolutions. Default is False.
        bn_kwargs (dict, optional): Additional keyword arguments for BatchNorm. Default is None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None): 
        super().__init__() 

        # check arguments 
        if bn_kwargs is None: 
            bn_kwargs = {} 

        # pointwise 
        self.add_module("pw", torch.nn.Conv2d( 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=(1, 1), 
                stride=1, 
                padding=0, 
                dilation=1, 
                groups=1, 
                bias=False, 
        )) 

        # batchnorm 
        if with_bn: 
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs)) 

        # depthwise 
        self.add_module("dw", torch.nn.Conv2d( 
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                groups=out_channels, 
                bias=bias, 
                padding_mode=padding_mode, 
        )) 
