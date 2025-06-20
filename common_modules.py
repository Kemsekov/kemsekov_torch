from typing import List, Literal, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

# Channel-wise Layer Normalization for N'd inputs
class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(2)
        # Input x has shape (batch_size, dim, seq_len)
        # Transpose to (batch_size, seq_len, dim) for nn.LayerNorm
        x = x.transpose(1, 2)
        # Apply layer normalization over the last dimension (dim)
        x = self.ln(x)
        # Transpose back to (batch_size, dim, seq_len)
        x = x.transpose(1, 2)
        return x.view(x_shape)
class ConstModule(torch.nn.Module):
    """Module that returns constant"""
    def __init__(self,constant = 0):
        super().__init__()
        self.constant=constant
    def forward(self,x):
        return self.constant

# @torch.jit.script
def resize_tensor(input : torch.Tensor,output_size : List[int],dimension_resize_mode : str = 'nearest-exact',channel_resize_mode : str='nearest-exact'):
    """
    Resizes input 1d,2d,3d tensor to given size, all up to channels
    """
    output_size=list(output_size)
    is_unsqueeze = False
    if len(input.shape)==2:
        input=input.unsqueeze(1)
        output_size=[1]+list(output_size)
        is_unsqueeze=True
        
    if input.shape[1:]==torch.Size(output_size):
        return input
    
    dim_size = torch.Size(output_size[1:])
    if input.shape[2:]!=dim_size:
        resize_dim = nn.functional.interpolate(input,dim_size,mode=dimension_resize_mode).transpose(1,2)
    else:
        resize_dim = input.transpose(1,2)
    
    ch_size    = list(output_size[1:])
    ch_size[0] = output_size[0]
    ch_size=torch.Size(ch_size)
    if resize_dim.shape[2:]!=ch_size:
        resize_channel = nn.functional.interpolate(resize_dim,ch_size,mode=channel_resize_mode).transpose(1,2)
    else:
        resize_channel = resize_dim.transpose(1,2)
    
    if is_unsqueeze:
        return resize_channel[:,0,:]
    return resize_channel

class Resize(nn.Module):
    """
    A PyTorch module that adjusts the spatial dimensions and channel count of an input tensor by simply resizing it.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size=output_size

    def forward(self, x):
        return resize_tensor(x,self.output_size)

class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor=scale_factor
        
    def forward(self,x):
        shape =  x.shape
        shape = torch.Size(list(shape[:2])+[int(v*self.scale_factor) for v in shape[2:]])[1:]
        return resize_tensor(x,shape)

class Mean0Std1Norm(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self,x):
        dims = list(range(len(x.shape)))[2:]
        mean = x.mean(dims,keepdim=True)
        std = x.std(dims,keepdim=True)+1e-6
        return (x-mean)/std
class SymetricLog(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mask = x>0
        out = x*0
        out[mask]=(1+x[mask]).log().to(x.dtype)
        out[~mask]=-(1-x[~mask]).log().to(x.dtype)
        return out

class SymetricSqrt(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mask = x>0
        out = x*0
        out[mask]=((1+x[mask]).sqrt()-1).to(x.dtype)
        out[~mask]=(1-(1-x[~mask]).sqrt()).to(x.dtype)
        return out
def get_normalization_from_name(dimensions, normalization: Literal['batch', 'instance','layer', 'group','Mean0Std1', None]):
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
    
    norm_type = {
        "batch": [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dimensions - 1],
        "instance": [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][dimensions - 1],
        "layer": ChanLayerNorm,
        "mean0std1": Mean0Std1Norm,
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
    
    allowed = list(norm_type.keys())+[None]
    assert normalization in allowed, f"normalization parameter must be one of {allowed}"
    
    if normalization is None:
        return nn.Identity
    
    return norm_type[normalization]

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
            
def reinit_with_ema(module, decay=0.99):
    """
    Slightly reinitializes the parameters of the given PyTorch module using EMA with the specified decay.
    
    Args:
        module (torch.nn.Module): The input module to be slightly reinitialized.
        decay (float): The decay factor for EMA. Default is 0.99.
    """
    # Create a deep copy of the module to preserve its structure
    orig_params = []
    for c in module.parameters():
        orig_params.append(c.clone())
        
    # Reinitialize parameters in the copied module
    for m in module.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    
    # Iterate over parameters and update using EMA
    for param, orig_param in zip(module.parameters(), orig_params):
        param.data = decay * orig_param.data + (1 - decay) * param.data