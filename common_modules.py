from typing import List, Literal, Tuple
import torch
import torch.nn as nn
class ConcatTensors(torch.nn.Module):
    """
        This module accepts list of tensors and concatenates them along axis `dim`
    """
    def __init__(self, dim = 1):
        """
        This module accepts list of tensors and concatenates them along axis `dim`
        """
        super().__init__()
        self.dim = dim
    def forward(self,tensors : List[torch.Tensor]):
        """
        tensors: a list of `torch.Tensor` objects that need to be concatenated
        """
        return torch.concat(tensors,self.dim)
class Residual(torch.nn.Module):
    """
    Residual module that sums outputs of module with it's input. It supports any models that outputs any shape.
    """
    def __init__(self,m : torch.nn.Module | List[torch.nn.Module],init_at_zero = True):
        """
        Residual module that wraps around module `m`.
        
        This module uses Re-Zero approach to add module output(multiplied by `alpha`) with it's inputs.
        
        When module output shape != input tensor shape, it uses nearest-exact resize approach to match input shape to output, 
        and performs addition as described.
        
        m: torch module, or list of modules (will be converted to sequential)
        init_at_zero: init module with learnable parameter 0.0, so only skip connection is present
        """
        super().__init__()
        if isinstance(m,list) or isinstance(m,tuple):
            m = torch.nn.Sequential(*m)
        
        self.m = m
        if init_at_zero:
            self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = 1
            
    def forward(self,x):
        out = self.m(x)
        if out.shape!=x.shape:
            x = resize_tensor(x,out.shape[1:])
        return self.alpha*out+x

class Repeat(torch.nn.Module):
    """
    Recursively repeats module output on itself
    """
    def __init__(self, m,repeats,grad_last_only = False):
        """
        m: module that must accept and return tensor of same shape
        repeats: how many time repeat the cycle
        grad_last_only: track gradients only on last iteration
        """
        super().__init__()
        assert repeats>0,f"repeats must be positive, got repeats={repeats}"
        self.repeats = repeats
        self.m=m
        self.grad_last_only=grad_last_only
    def forward(self,x):
        if self.grad_last_only:
            with torch.no_grad():
                for i in range(self.repeats-1):
                    x=self.m(x)
        else:
            for i in range(self.repeats-1):
                x=self.m(x)
        return self.m(x)

class AddConst(nn.Module):
    """Adds constant `c` to tensor"""
    def __init__(self,c):
        super().__init__()
        self.c = c
    def forward(self,x):
        return x+self.c

def _reshape_to_transformer_input(x : torch.Tensor):
    """
    x of shape [batch,channels,...dims...]
    """
    return x.flatten(2).permute(0,2,1)
def _restore_shape_of_transformer_output(out,src_shape : List[int]):
    return out.permute(0,2,1).view(src_shape)
class FlattenSpatialDimensions(nn.Module):
    """
    Prepares vison-like 1d,2d,3d sequential data into format suitable for transformer
    
    Permutes spatial dimension-like input 
    `[batch,channels,dim1,dim2,...]` to `[batch,dim*dim2*...,channels]`
    
    Then feeds this tensor to input module m and reshapes it's output back to original shape.
    """
    def __init__(self, m):
        """
        Permutes spatial dimension-like input 
        `[batch,channels,dim1,dim2,...]` to `[batch,dim*dim2*...,channels]`
        
        Then feeds this tensor to input module m and reshapes it's output back to original shape.

        m: `torch.nn.Module` or `List[torch.nn.Module]`
        """
        super().__init__()
        if isinstance(m,list) or isinstance(m,tuple):
            self.m = nn.Sequential(*m)
        else:
            self.m  = m
        
    def forward(self,x):
        x_shape = list(x.shape)
        x_flat = _reshape_to_transformer_input(x)
        out = self.m(x_flat)
        x_shape[1] = out.shape[-1] # update channels
        return _restore_shape_of_transformer_output(out,torch.Size(x_shape))
class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self,x):
        return x.unsqueeze(self.dim)
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self,x):
        return x.squeeze(self.dim)
class Take(nn.Module):
    def __init__(self, slice):
        super().__init__()
        self.slice=slice
    def forward(self,x):
        return x[self.slice]

class Permute(nn.Module):
    """
    Permutes input tensor with given permutation
    """
    def __init__(self, permutation):
        """
        Permutes input tensor with given permutation
        """
        super().__init__()
        self.permuitation = permutation
        
    def forward(self,x):
        return x.permute(self.permuitation)

class Transpose(nn.Module):
    """
    Transpose input tensor along given dims
    """
    def __init__(self, dim1,dim2):
        """
        Permutes input tensor with given permutation
        """
        super().__init__()
        self.dim1 = dim1
        self.dim2= dim2
        
    def forward(self,x):
        return x.transpose(self.dim1,self.dim2)


class Flatten(nn.Module):
    def __init__(self, start_dim,end_dim,submodule):
        """
        A wrapper module that flattens a subset of input tensor dimensions before applying a submodule,
        and then reshapes the output back to the original input shape.

        This is useful when you want to apply a submodule (e.g., Linear or LayerNorm) over a flattened
        region of a tensor (e.g., over spatial or temporal dimensions) while keeping the batch and channel
        structure intact.

        Args:
            start_dim (int): First dimension to flatten.
            end_dim (int): Last dimension to flatten (inclusive).
            submodule (nn.Module): The module to apply to the flattened tensor.

        Forward Input:
            x (Tensor): Arbitrary-shaped tensor.

        Forward Output:
            Tensor: Output of submodule, reshaped to original input shape.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        if isinstance(submodule,tuple) or isinstance(submodule,list):
            self.m = nn.Sequential(*submodule)
        else:
            self.m = submodule
    def forward(self,x):
        x_shape = x.shape
        x_flat = torch.flatten(x,self.start_dim,self.end_dim)
        y = self.m(x_flat)
        return y.reshape(x_shape)

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
class Interpolate(nn.Module):
    """
    Scales `[batch,channels,...]` tensor (...) dimensions by a factor of `scale_factor`
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor=scale_factor
        
    def forward(self,x):
        shape =  x.shape
        shape = torch.Size(list(shape[:2])+[int(v*self.scale_factor) for v in shape[2:]])[1:]
        return resize_tensor(x,shape)
class Resize(nn.Module):
    """
    A PyTorch module that adjusts the spatial dimensions and channel count of an input tensor by simply resizing it.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size=output_size

    def forward(self, x):
        return resize_tensor(x,self.output_size)

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
class Mean0Std1Norm(torch.nn.Module):
    """
    Transforms input of shape [Batch,Channels,...] to have mean 0 std 1 along spatial dimensions (...)
    """
    def __init__(self):
        super().__init__()
    def forward(self,x):
        dims = list(range(len(x.shape)))[2:]
        mean = x.mean(dims,keepdim=True)
        std = x.std(dims,keepdim=True)+1e-6
        return (x-mean)/std

def module_params_count(module : nn.Module):
    """
    Returns count of module parameters
    """
    return sum([n.numel() for n in module.parameters()])
    

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
            setattr(module,d,wrapper(el))
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


def smooth_abs(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.abs(x)
    return torch.where(abs_x > 1, abs_x, 0.5 * x**2 + 0.5)

def kl_divergence(mu : torch.Tensor, logvar : torch.Tensor ,latent_dimension=[-1]):
    """
    Compute KL divergence between N(mu, sigma^2) and N(0, 1).
    
    Args:
        mu (torch.Tensor): Mean of the learned distribution
        logvar (torch.Tensor): Log standard deviation of the learned distribution
        latent_dimension int: index of dimensions along which to compute sum
    
    Returns:
        torch.Tensor: KL divergence loss
    """
    
    # KL divergence per element
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Reduce along latent dimensions
    if latent_dimension is not None:
        kl = kl.sum(dim=latent_dimension)
    
    return kl.mean()


def mmd_rbf(A, B, gamma=None, eps=1e-12):
    """
    Compute unbiased MMD^2 between samples A and B using an RBF (Gaussian) kernel.

    Args:
        A: array-like, shape (n, d)
        B: array-like, shape (m, d)
        gamma: RBF parameter. If None, uses median pairwise-distance heuristic.
               Kernel is k(x,y)=exp(-gamma * ||x-y||^2). [web:336]
        eps: numerical stability

    Returns:
        mmd2: float, unbiased estimate of MMD^2 [web:334][web:336]
        gamma: float, kernel parameter used
    """
    if not isinstance(A,torch.Tensor):
        A = torch.tensor(A)
    if not isinstance(B,torch.Tensor):
        B = torch.tensor(B)
    n, d = A.shape
    m, d2 = B.shape
    assert d == d2

    def sq_dists(X, Y):
        XX = torch.sum(X * X, axis=1)[:, None]
        YY = torch.sum(Y * Y, axis=1)[None, :]
        return XX + YY - 2.0 * (X @ Y.T)

    if gamma is None:
        Z = torch.vstack([A, B])
        D = sq_dists(Z, Z)
        tri = D[torch.triu_indices(D.shape[0], col=1)]
        med = torch.median(tri)
        gamma = 5 / (med + eps)  # simple median heuristic choice

    Kxx = torch.exp(-gamma * sq_dists(A, A))
    Kyy = torch.exp(-gamma * sq_dists(B, B))
    Kxy = torch.exp(-gamma * sq_dists(A, B))

    # unbiased: remove diagonal self-similarity terms in Kxx and Kyy [web:334][web:336]
    Kxx.fill_diagonal_(0.0)
    Kyy.fill_diagonal_(0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1))) + (Kyy.sum() / (m * (m - 1))) - 2.0 * Kxy.mean()
    return mmd2, gamma

from torch.autograd import Function
class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # Прямой проход оставляет тензор неизменным
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # В обратном проходе умножаем градиент на -lambda
        return grad_output.neg() * ctx.lambda_, None

class GradientReversal(nn.Module):
    """
    A Gradient Reversal Layer.

    During the forward pass, acts as an identity function:
        y = x

    During the backward pass, reverses the gradient by multiplying it by -lambda:
        dL/dx = -lambda * dL/dy

    Args:
        lambda_ (float): The scaling factor for reversed gradients. Default is 1.0.

    Example:
        >>> grl = GradientReversal(lambda_=0.5)
        >>> x = torch.randn(16, 128, requires_grad=True)
        >>> y = grl(x)
        >>> # Pass y through a classifier head to obtain `logits`
        >>> logits = classifier(y)
        >>> loss = loss_fn(logits, targets)
        >>> loss.backward()
        >>> # Now x.grad will have been multiplied by -0.5
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)


import torch.utils.checkpoint as checkpoint
class OffloadWrapper(nn.Module):
    def __init__(
        self, 
        base_module,        
        offload_device='cpu', 
        compute_device='cuda',
    ) -> None:
        super().__init__()
        self.base_module=base_module
        self.offload_device=offload_device
        self.compute_device=compute_device
        
    def forward(self,x):
        not_same = not self.compute_device==self.offload_device
        if not_same:
            self.base_module.to(self.compute_device, non_blocking=True)
        x = self.base_module(x)
        if not_same:
            self.base_module.to(self.offload_device, non_blocking=True)
        return x

class CheapSequential(nn.Module):
    """
    A memory-efficient sequential container that allows for automatic checkpointing 
    and offloading of intermediate modules to reduce GPU memory usage.

    This module is similar to `torch.nn.Sequential` but designed for very large
    models where holding all modules in GPU memory simultaneously may be impractical. 
    Each submodule is wrapped in an `OffloadWrapper` that temporarily moves the 
    module to the GPU for computation and offloads it back to CPU after the forward 
    pass. Optionally, gradient checkpointing is applied to save memory during backpropagation.

    Parameters
    ----------
    modules : list[nn.Module]
        A list of PyTorch modules that form the sequential model. Each module 
        will be individually wrapped for offloading.

    offload_device : str or torch.device, default='cpu'
        The device to offload modules to after computation. Typically 'cpu'.

    compute_device : str or torch.device, default='cuda'
        The device to perform computation on (forward pass). Typically 'cuda'.

    Attributes
    ----------
    m : nn.ModuleList
        List of `OffloadWrapper` modules, each handling its own offloading.
        
    compute_device : torch.device
        The device used to perform forward computation.

    Methods
    -------
    forward(input)
        Performs a forward pass through the sequence of modules. Each module is
        temporarily moved to the `compute_device` for computation, and offloaded
        to `offload_device` after use. If gradients are enabled, checkpointing is 
        applied for memory efficiency.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> from my_module import CheapSequential
    >>> 
    >>> layers = [nn.Linear(1024, 1024) for _ in range(5)]
    >>> model = CheapSequential(layers, offload_device='cpu', compute_device='cuda')
    >>> x = torch.randn(2, 1024, device='cuda')
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 1024])
    
    Notes
    -----
    - The `forward` method must not change the device of any tensor that requires 
      gradients in the middle of backpropagation. Checkpointing moves inputs through 
      submodules without retaining the module on GPU.
    - Moving submodules inside `forward` may break autograd if parameters are moved 
      before backward finishes. The current implementation works with checkpointing.
    - This class is mainly useful for very large models where GPU memory is limited.
    """
    def __init__(
        self,
        modules,
        offload_device='cpu', 
        compute_device='cuda',
    ):
        super().__init__()
        self.m=nn.ModuleList([
            OffloadWrapper(m,offload_device,compute_device) 
            for m in modules
        ]).to(offload_device, non_blocking=True)
        self.compute_device=compute_device
        
    def forward(self, input):
        input_device = input.device
        x = input.to(self.compute_device)
        for m in self.m:
            if torch.is_grad_enabled():
                def _run(input_tensor, module=m):
                    # clone to protect against in-place ops inside module
                    return module(input_tensor.clone())

                x = checkpoint.checkpoint(_run,x,use_reentrant=False)
            else:
                x = m(x)
        return x.to(input_device)