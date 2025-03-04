from typing import Literal
import torch
import torch.nn.functional as F
import torch.nn as nn


class ConstModule(torch.nn.Module):
    """Module that returns constant"""
    def __init__(self,constant = 0):
        super().__init__()
        self.constant=constant
    def forward(self,x):
        return self.constant

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
    def __init__(self, in_ch, out_ch, scale_factor, dimensions=2, mode='bilinear',normalization='batch'):
        super(UpscaleResize, self).__init__()
        if dimensions not in (1, 2, 3):
            raise ValueError("dimensions must be 1, 2, or 3")

        self.dimensions = dimensions
        self.scale_factor = float(scale_factor)
        self.mode=mode
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
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
            
        # Adjust the number of channels
        x = self.channel_adjust(x)

        return x


def get_normalization_from_name(dimensions, normalization: Literal['batch', 'instance', 'group', None]):
    """Get normalization for given dimensions from its name.

    Args:
        dimensions (int): Dimensionality of the input tensor (1, 2, or 3).
        normalization (Literal['batch', 'instance', 'group', None]): Type of normalization to apply.

    Returns:
        callable: A normalization module constructor based on the specified type and dimensions.
                  For 'group', dynamically determines `num_groups` based on channel count.

    Raises:
        AssertionError: If `normalization` is not one of ['batch', 'instance', 'group', None].
    """
    allowed = ['batch', 'instance', 'group', None]
    assert normalization in allowed, f"normalization parameter must be one of {allowed}"
    
    norm_type = {
        "batch": [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dimensions - 1],
        "instance": [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][dimensions - 1],
        "group": lambda ch: nn.GroupNorm(
            num_groups=(
                ch // 32 if ch % 32 == 0 and ch // 32 >= 2 else
                ch // 16 if ch % 16 == 0 and ch // 16 >= 2 else
                ch // 8 if ch % 8 == 0 and ch // 8 >= 2 else
                ch // 4 if ch % 4 == 0 and ch // 4 >= 2 else
                ch
            ),
            num_channels=ch
        )
    }
    
    if normalization is None:
        return nn.Identity
    
    return norm_type[normalization]

def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).
    """
    while b:
        a, b = b, a%b
    return a

def wrap_submodules(module,module_type,wrapper):
    """
    Applies wrapper to module and/or all it's submodules, that matches
    `module_type`
    """
    # handle list/nn.Sequential/nn.ModuleList
    
    try:
        for i in range(len(module)):
            el = module[i]
            module[i] = el
        is_set_iterable = True
    except Exception as e:
        is_set_iterable = False
        pass
    if is_set_iterable:
        for i in range(len(module)):
            el = module[i]
            if isinstance(el,module_type):
                module[i]=wrapper(el)
                continue
            if isinstance(el,torch.nn.Module):
                wrap_submodules(el,module_type,wrapper)
    # handle dictionary-like types
    try:
        for key in module:
            el = module[key]
            module[key]=wrapper(el)
        is_set_dict = True
    except: 
        is_set_dict = False
        pass
    if is_set_dict:
        for key in module:
            el = module[key]
            if isinstance(el,module_type):
                module[key]=wrapper(el)
                continue
            wrap_submodules(el,module_type,wrapper)
    for d in dir(module):
        if not hasattr(module,d): continue
        el = getattr(module,d)
        if isinstance(el,module_type):
            setattr(module,d,wrap_submodules(el,wrapper))
            continue
        if isinstance(el,torch.nn.Module):
            wrap_submodules(el,module_type,wrapper)