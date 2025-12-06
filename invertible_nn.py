from typing import Union
import torch
import torch.nn as nn

class InvertibleLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        # y = x if x >= 0 else negative_slope * x
        return torch.where(x >= 0, x, self.negative_slope * x)

    def inverse(self, y):
        # x = y if y >= 0 else y / negative_slope
        return torch.where(y >= 0, y, y / self.negative_slope)

    def derivative(self, x):
        # d/dx = 1 if x >= 0 else negative_slope
        return torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, self.negative_slope))

class InvertibleTanh(torch.nn.Module):
    def __init__(self,scale=2):
        super().__init__()
        self.scale=scale
    
    def forward(self, x):
        return self.scale*torch.tanh(x)
    
    def inverse(self, y):
        # Inverse: y = scale * tanh(x) => x = arctanh(y/scale)
        # arctanh(z) = 0.5 * ln((1+z)/(1-z)), where z = y/scale
        z = y / self.scale
        # Clamp z to [-1+eps, 1-eps] to avoid numerical instability
        z = torch.clamp(z, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return 0.5 * torch.log((1 + z) / (1 - z))
    
    def derivative(self, x):
        # Derivative: d/dx [scale * tanh(x)] = scale * (1 - tanh^2(x))
        return self.scale * (1 - torch.tanh(x)**2)

class SymmetricLog(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mask = x>0
        out = x*0
        out[mask]=(1+x[mask]).log().to(x.dtype)
        out[~mask]=-(1-x[~mask]).log().to(x.dtype)
        return out
    
    def inverse(self, y):
        mask = y >= 0
        out = torch.empty_like(y)
        # positive branch: y = log(1+x)  ⇒  x = exp(y) - 1
        out[mask] = torch.exp(y[mask]) - 1
        # negative branch: y = -log(1-x) ⇒  x = 1 - exp(-y)
        out[~mask] = 1 - torch.exp(-y[~mask])
        return out
    
    def derivative(self, x):
        mask = x > 0
        out = torch.empty_like(x)
        # for x > 0: d/dx [log(1+x)] = 1/(1+x)
        out[mask] = 1.0 / (1 + x[mask] + 1e-6)
        # for x <= 0: d/dx [-log(1-x)] = 1/(1-x)
        out[~mask] = 1.0 / (1 - x[~mask] + 1e-6)
        return out

class SymmetricSqrt(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mask = x>0
        out = x*0
        out[mask]=((1+x[mask]).sqrt()-1).to(x.dtype)
        out[~mask]=(1-(1-x[~mask]).sqrt()).to(x.dtype)
        return out
    def inverse(self, y):
        mask = y >= 0
        out = torch.empty_like(y)
        out[mask]   = (y[mask] + 1).pow(2) - 1
        out[~mask]  = 1 - (1 - y[~mask]).pow(2)
        return out
    
    def derivative(self, x):
        mask = x > 0
        out = torch.empty_like(x)
        # for x>0: d/dx [sqrt(1+x)-1] = 1/(2*sqrt(1+x))
        out[mask] = 1.0 / (2 * (1 + x[mask]).sqrt()+1e-6)
        # for x<=0: d/dx [1 - sqrt(1-x)] = 1/(2*sqrt(1-x))
        out[~mask] = 1.0 / (2 * (1 - x[~mask]).sqrt()+1e-6)
        return out

import torch
import torch.nn as nn
from torch.autograd import Function

class _SmoothSymmetricLogFunction(Function):
    """
    Custom autograd function that uses the exact derivative in backward pass
    while keeping the forward approximation.
    """
    @staticmethod
    def forward(ctx, x, approx_module):
        """
        Forward pass: use the approximation
        """
        ctx.save_for_backward(x)
        ctx.approx_module = approx_module
        return approx_module._compute(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: use the exact derivative instead of backpropagating through forward approximation
        """
        x, = ctx.saved_tensors
        approx_module = ctx.approx_module
        # print("back",x)
        # Compute exact derivative: 1/(|x| + e^(-|x|))
        x_abs = x.abs()
        exact_derivative = 1.0 / (x_abs + torch.exp(-x_abs))
        
        # Handle sign for negative x values (since integral is odd function)
        sign = torch.sign(x)
        sign[sign == 0] = 1  # Handle x=0 case
        
        # Apply chain rule: grad_input = grad_output * derivative
        grad_input = grad_output * exact_derivative * sign
        
        return grad_input, None  # None for approx_module (not differentiable)

class SmoothSymmetricLog3rdOrder(nn.Module):
    def __init__(self, split_point=8.4211):
        super().__init__()
        params=[
            0.9926349357766904,
            0.8110374873908223,
            0.10474537829412185,
            0.7690230856892495,
            0.3516600050453235,
            0.0199296372515136,
            0.00010754123262324145
        ]
        # Coefficients for 3rd order rational approximation (0 <= x <= 8)
        self.register_buffer('a1', torch.tensor(params[0]))
        self.register_buffer('a2', torch.tensor(params[1]))
        self.register_buffer('a3', torch.tensor(params[2]))
        self.register_buffer('b1', torch.tensor(params[3]))
        self.register_buffer('b2', torch.tensor(params[4]))
        self.register_buffer('b3', torch.tensor(params[5]))
        self.register_buffer('num_const', torch.tensor(params[6]))
        
        # Constant for logarithmic approximation (x > 8)
        self.log_const = 0.767229259388
        self.split_point = split_point
    
    def forward(self, x):
        """
        Approximate F(x) = ∫₀ˣ 1/(t + e^(-t)) dt with custom backward pass
        """
        # Handle negative values using odd function property: F(-x) = -F(x)
        x_g0 = x > 0
        x_s0 = ~x_g0
        
        # Create output tensor
        out = torch.zeros_like(x)
        # Positive values: compute directly
        if torch.any(x_g0):
            out[x_g0] = _SmoothSymmetricLogFunction.apply(x[x_g0], self)
        
        # Negative values: use F(-x) = -F(x)
        if torch.any(x_s0):
            out[x_s0] = -_SmoothSymmetricLogFunction.apply(-x[x_s0], self)
        
        return out
    
    def _compute(self, x):
        """
        Core computation for x >= 0
        """
        # Create output tensor with same device and dtype as input
        out = torch.zeros_like(x)
        
        # Create masks
        mask_small = x <= self.split_point
        mask_large = ~mask_small
        
        # Compute rational approximation for x <= split_point
        if torch.any(mask_small):
            x_small = x[mask_small]
            
            # Numerator: a1*x + a2*x^2 + a3*x^3 + num_const
            numerator = (
                self.a1 * x_small + self.num_const +
                self.a2 * x_small**2 +
                self.a3 * x_small**3
            )
            
            # Denominator: 1 + b1*x + b2*x^2 + b3*x^3
            denominator = (
                1.0 +
                self.b1 * x_small +
                self.b2 * x_small**2 +
                self.b3 * x_small**3
            )
            
            out[mask_small] = numerator / denominator
        
        # Compute logarithmic approximation for x > split_point
        if torch.any(mask_large):
            x_large = x[mask_large]
            out[mask_large] = torch.log(x_large) + self.log_const
        
        return out
    
    def derivative(self, x):
        """
        Exact analytic derivative (for reference/debugging)
        """
        x_abs = x.abs()
        return 1.0 / (x_abs + torch.exp(-x_abs))


class SmoothSymmetricLog4rdOrder(nn.Module):
    def __init__(self,split_point = 8.1044):
        super().__init__()
        
        params=[0.9998109410074897,
        0.8425166425498785,
        0.412532839718835,
        0.02168303733037949,
        0.8424443481738484,
        0.5758277939280647,
        0.13134256599296878,
        0.0037374247639141932,
        8.936675541995365e-06]
        
        # Coefficients for rational approximation (0 <= x <= 8)
        # Using nn.Parameter to allow potential fine-tuning if needed
        self.register_buffer('a1', torch.tensor(params[0]))
        self.register_buffer('a2', torch.tensor(params[1]))
        self.register_buffer('a3', torch.tensor(params[2]))
        self.register_buffer('a4', torch.tensor(params[3]))
        self.register_buffer('b1', torch.tensor(params[4]))
        self.register_buffer('b2', torch.tensor(params[5]))
        self.register_buffer('b3', torch.tensor(params[6]))
        self.register_buffer('b4', torch.tensor(params[7]))
        self.register_buffer('num_const', torch.tensor(params[8]))
        
        
        # Constant for logarithmic approximation (x > 8)
        self.log_const = 0.767229259388
        self.split_point = split_point
    
    def forward(self, x):
        """
        Approximate F(x) = ∫₀ˣ 1/(t + e^(-t)) dt with custom backward pass
        """
        # Handle negative values using odd function property: F(-x) = -F(x)
        x_g0 = x > 0
        x_s0 = ~x_g0
        
        # Create output tensor
        out = torch.zeros_like(x)
        
        # Positive values: compute directly using custom autograd
        if torch.any(x_g0):
            out[x_g0] = _SmoothSymmetricLogFunction.apply(x[x_g0], self)
        
        # Negative values: use F(-x) = -F(x)
        if torch.any(x_s0):
            out[x_s0] = -_SmoothSymmetricLogFunction.apply(-x[x_s0], self)
        
        return out
    
    def _compute(self, x):

        # Create output tensor with same device and dtype as input
        out = torch.zeros_like(x)
        
        # Create masks
        mask_small = x <= self.split_point
        mask_large = ~mask_small
        
        # Compute rational approximation for x <= 8
        if torch.any(mask_small):
            x_small = x[mask_small]
            
            # Numerator: a1*x + a2*x^2 + a3*x^3 + a4*x^4
            numerator = (
                self.a1 * x_small + self.num_const+
                self.a2 * x_small**2 +
                self.a3 * x_small**3 +
                self.a4 * x_small**4
            )
            
            # Denominator: 1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
            denominator = (
                1.0 +
                self.b1 * x_small +
                self.b2 * x_small**2 +
                self.b3 * x_small**3 +
                self.b4 * x_small**4
            )
            
            out[mask_small] = numerator / denominator
        
        # Compute logarithmic approximation for x > 8
        if torch.any(mask_large):
            x_large = x[mask_large]
            out[mask_large] = torch.log(x_large) + self.log_const
        
        return out
    
    def derivative(self,x):
        x_abs = x.abs()
        return 1.0/(x_abs+torch.exp(-x_abs))


def flip_even_odd(x: torch.Tensor, dim: int = -1):
    """
    Split tensor into even- and odd-indexed slices along a given dimension.

    Args:
        x: input tensor
        dim: dimension along which to split

    Returns:
        even, odd (two tensors with the same shape except along dim)
    """
    # build index slices
    even_idx = torch.arange(0, x.size(dim), 2, device=x.device)
    odd_idx = torch.arange(1, x.size(dim), 2, device=x.device)

    even = torch.index_select(x, dim, even_idx)
    odd = torch.index_select(x, dim, odd_idx)

    return torch.concat([even, odd],dim)

def unflip_even_odd(x: torch.Tensor, dim: int = -1):
    """
    Undo flip_even_odd: interleave even and odd parts back along a given dim.

    Args:
        x: tensor produced by flip_even_odd
        dim: dimension along which to invert

    Returns:
        Tensor with original ordering restored
    """
    n = x.size(dim)
    half = n // 2 + n % 2  # number of evens (handle odd length safely)

    # split back into even and odd chunks
    even, odd = torch.split(x, [half, n - half], dim=dim)

    # create empty tensor to hold result
    out = torch.empty_like(x)

    # scatter even indices
    out.index_copy_(dim, torch.arange(0, n, 2, device=x.device), even)
    # scatter odd indices
    out.index_copy_(dim, torch.arange(1, n, 2, device=x.device), odd)

    return out

class InvertibleScaleAndTranslate(nn.Module):
    """
    Invertible neural network for normalizing flows, applying scaling, translation, and shuffling with nonlinear function, which provides infinitely differentiable Invertible neural network.
    
    Args:
        model (nn.Module): Neural network to compute scaling and translation factors. It takes input with half dimensions along specified dim and returns twice of it.
        dimension_split (int, optional): Dimension to split the input. Defaults to -1 (last dimension).
        non_linearity (torch.nn.Module): invertible non-linearity function that is used to improve model expressiveness
    """
    def __init__(self, model,dimension_split = -1,non_linearity : Union[InvertibleTanh,SymmetricSqrt,SymmetricLog,InvertibleLeakyReLU] = InvertibleTanh(2)):
        super().__init__()
        self.model=model
        self.dimension_split = dimension_split  # Ensure integer type
        self.non_linearity=non_linearity
    
    def get_scale_and_translate(self,x):
        scale,translate = self.model(x).chunk(2,self.dimension_split)
        # make scale positive
        scale=torch.nn.functional.elu(scale)+1
        return scale,translate
    
    def forward(self, input):
        """
        Forward transformation: split, scale/translate, concatenate, and shuffle.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, ..., feature_dim).
        Returns:
            Tuple[torch.Tensor,torch.Tensor]: 
                Transformed tensor with the same shape as input and scale parameter.
        """
        x1,x2 = input.chunk(2,self.dimension_split)
        scale,translate = self.get_scale_and_translate(x1)
        
        z2 = x2*scale+translate
        concat = torch.concat([self.non_linearity(z2),x1],self.dimension_split)
        concat = flip_even_odd(concat,self.dimension_split)
        
        jacob_det = self.non_linearity.derivative(z2)*scale
        
        return concat, jacob_det

    def inverse(self,output):
        """
        Inverse transformation: deshuffle, split, invert scale/translate, and concatenate.
        
        Args:
            output (torch.Tensor): Output tensor from forward pass.
        
        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        output = unflip_even_odd(output,self.dimension_split)
        
        f_z2,x1 = output.chunk(2,self.dimension_split)
        z2 = self.non_linearity.inverse(f_z2)
        
        scale,translate = self.get_scale_and_translate(x1)
        x2 = (z2-translate)/(scale+1e-6)
        concat = torch.concat([x1,x2],self.dimension_split)
        return concat
  
class InvertibleSequential(nn.Sequential):
    """
    Sequential container for invertible modules, supporting forward and inverse transformations.
    
    Args:
        *modules: Sequence of invertible nn.Module instances.
    """
    def __init__(self, *modules):
        super().__init__(*modules)
        
    def forward(self,input):
        """
        Computes forward pass of Invertible neural networks composition.
        
        returns:
        output and list of jacobians determinants
        """
        out = input
        jacobians = []
        for m in self:
            out,jacob = m(out)
            jacobians.append(jacob)
        
        return out,jacobians
    
    def inverse(self,out):
        """
        Applies inverse transformations of all modules in reverse order.
        
        Args:
            out (torch.Tensor): Output tensor from forward pass.
        
        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        prev = out
        for m in reversed(self):
            prev = m.inverse(prev)
        return prev
        
