from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import *
    
class BSConvU(torch.nn.Module): 
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
        dimensions (int,optional): Convolution dimensions, must be one of [1,2,3]
        with_bn (bool, optional): If True, includes BatchNorm between pointwise and depthwise convolutions. Default is False.
        bn_kwargs (dict, optional): Additional keyword arguments for BatchNorm. Default is None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros",dimensions=2, with_bn=False, bn_kwargs=None): 
        super().__init__() 
        assert dimensions in [1,2,3], f"{dimensions} must be in range [1,2,3]"
        self.conv = [BSConvU1d,BSConvU2d,BSConvU3d][dimensions-1](
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            bias=bias, 
            padding_mode=padding_mode,
            with_bn=with_bn, 
            bn_kwargs=bn_kwargs
        )
    def forward(self,x):
        return self.conv(x)

class BSConvU1d(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()

        if bn_kwargs is None:
            bn_kwargs = {}

        self.add_module("pw", torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm1d(num_features=out_channels, **bn_kwargs))

        self.add_module("dw", torch.nn.Conv1d(
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

class BSConvU2d(torch.nn.Sequential): 
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

class BSConvU3d(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()

        if bn_kwargs is None:
            bn_kwargs = {}

        self.add_module("pw", torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm3d(num_features=out_channels, **bn_kwargs))

        self.add_module("dw", torch.nn.Conv3d(
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
