from typing import Literal
import torch
import torch.nn.functional as F
import torch.nn as nn

# change tensor shape
class Interpolate(torch.nn.Module):
    def __init__(self, scale_factor=None, size=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode, align_corners=self.align_corners)

class UpscaleResize(nn.Module):
    """
    A PyTorch module that adjusts the spatial dimensions and channel count of an input tensor.

    This module performs the following operations:
    - **Spatial Resizing:** If `scale_factor` is not 1, the input tensor's spatial dimensions are scaled by the specified factor using nearest-neighbor interpolation.
    - **Channel Adjustment:** If `in_ch` differs from `out_ch`, a 1x1 convolution appropriate for the specified dimensions is applied to modify the number of channels. If `in_ch` equals `out_ch`, this step is bypassed to enhance efficiency.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Desired number of output channels.
        scale_factor (float): Factor by which to scale the spatial dimensions of the input tensor.
        dimensions (int): Dimensionality of the input tensor (1, 2, or 3).

    Example:
        >>> upscale_resize = UpscaleResize(in_ch=64, out_ch=128, scale_factor=2, dimensions=2)
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> output_tensor = upscale_resize(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 128, 64, 64])
    """
    def __init__(self, in_ch, out_ch, scale_factor, dimensions=2, mode='nearest',normalization='batch'):
        super(UpscaleResize, self).__init__()
        if dimensions not in (1, 2, 3):
            raise ValueError("dimensions must be 1, 2, or 3")

        self.dimensions = dimensions
        self.scale_factor = float(scale_factor)

        # Determine if channel adjustment is necessary
        if in_ch == out_ch:
            self.channel_adjust = nn.Identity()
        else:
            if dimensions == 1:
                self.channel_adjust = nn.Sequential(
                    nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                    get_normalization_from_name(dimensions,normalization)(out_ch)
                )
            elif dimensions == 2:
                self.channel_adjust = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                    get_normalization_from_name(dimensions,normalization)(out_ch)
                )
            elif dimensions == 3:
                self.channel_adjust = nn.Sequential(
                    nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
                    get_normalization_from_name(dimensions,normalization)(out_ch)
                )

    def forward(self, x):
        # Apply spatial resizing only if scale_factor is not 1
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
            
        # Adjust the number of channels
        x = self.channel_adjust(x)

        return x


def get_normalization_from_name(dimensions,normalization:Literal['batch','instance','group',None]):
    """Get normalization for given dimensions from it's name"""
    allowed = ['batch','instance','group',None]
    assert normalization in allowed, f"normalization parameter must be one of {allowed}"
    norm_type = {
            "batch":[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d][dimensions-1],
            "instance":[nn.InstanceNorm1d,nn.InstanceNorm2d,nn.InstanceNorm3d][dimensions-1],
            "group": lambda ch: nn.GroupNorm(ch//8,ch)
        }
    
    if normalization is None:
        return nn.Identity
    
    return norm_type[normalization]