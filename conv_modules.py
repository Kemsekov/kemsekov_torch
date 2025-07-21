import torch
import torch.nn as nn
import torch.nn.functional as F
from kemsekov_torch.common_modules import get_normalization_from_name




class BSConvU(torch.nn.Sequential): 
    """
    Blueprint Pointwise-Depthwise Convolution Block.

    This block implements a convolutional module that performs pointwise convolution
    followed by depthwise convolution. It is an alternative to the standard depthwise-pointwise 
    approach and may be more effective in certain use cases.

    Attributes:
        pw (torch.nn.Conv[1,2,3]d): The pointwise convolution (1x1) layer.
        gn (torch.nn.GroupNorm, optional): Optional GroupNorm applied after the pointwise convolution.
        dw (torch.nn.Conv[1,2,3]d): The depthwise convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the depthwise convolution kernel.
        stride (int, optional): Stride for the depthwise convolution. Default is 1.
        padding (int or tuple, optional): Implicit padding for the depthwise convolution. Default is 0.
        dilation (int, optional): Dilation rate for the depthwise convolution. Default is 1.
        bias (bool, optional): Whether to add a learnable bias to the depthwise convolution. Default is True.
        padding_mode (str, optional): Padding mode for the depthwise convolution. Default is 'zeros'.
        dimensions (int,optional): Convolution dimensions, must be one of [1,2,3]
        with_bn (bool, optional): If True, includes BatchNorm between pointwise and depthwise convolutions. Default is False.
        bn_kwargs (dict, optional): Additional keyword arguments for BatchNorm. Default is None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros",dimensions=2, with_gn=False, bn_kwargs=None,transpose=False): 
        super().__init__() 

        if bn_kwargs is None:
            bn_kwargs = {}
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        self.pw= conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        if with_gn:
            self.gn = get_normalization_from_name(dimensions,'group')(out_channels)
        else:
            self.gn=nn.Identity()
        if transpose:
            conv = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][dimensions-1]

        self.dw = conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
    def forward(self,x):
        x = self.pw(x)
        x=self.gn(x)
        return self.dw(x)
        
class BSConvUTranspose(BSConvU):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros",dimensions=2, with_gn=False, bn_kwargs=None): 
        super().__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            bias, 
            padding_mode,
            dimensions, 
            with_gn, 
            bn_kwargs,
            transpose=True
        ) 