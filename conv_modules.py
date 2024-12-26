import torch
import torch.nn as nn

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
