import random
from typing import Union
import math
import torch.nn as nn
import torch

def _test_invertible_module(m : nn.Module,x0 = -5,x1 = 5):
    x = torch.linspace(x0,x1,1024).requires_grad_(True)
    y = m(x)
    y.sum().backward()
    xgrad = x.grad
    xderiv = m.derivative(x)
    yinv = m.inverse(y)
    
    print("Inverse MAE",(x-yinv).abs().mean().item())
    print("deriv MAE",(xderiv-xgrad).abs().mean().item())

class InvertibleComposition(nn.Module):
    """
    Composition of two invertible modules:
        y = g(f(x))

    Assumes each module implements:
      - forward(x) -> y
      - inverse(y) -> x
      - derivative(x) -> dy/dx   (elementwise diagonal factor, same shape as x)

    Then by chain rule:
      d/dx g(f(x)) = g'(f(x)) * f'(x)  (elementwise product) [web:296][web:297]

    Inverse of composition:
      (g ∘ f)^{-1} = f^{-1} ∘ g^{-1}  [web:297]
    """
    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x: torch.Tensor):
        y1 = self.f(x)
        y2 = self.g(y1)
        return y2

    def inverse(self, y: torch.Tensor):
        x1 = self.g.inverse(y)
        x0 = self.f.inverse(x1)
        return x0

    def derivative(self, x: torch.Tensor):
        # chain rule: g'(f(x)) * f'(x) [web:296]
        is_train = self.f.training
        y1 = self.f.eval()(x)
        if is_train:
            self.f.train()
            
        return self.g.derivative(y1) * self.f.derivative(x)

class KFunction(torch.nn.Module):
    def __init__(self, n_mid=11):
        super().__init__()
        self.n_mid = n_mid
        # Register core constants as buffers (will be moved with .to())
        self.register_buffer('pi', torch.tensor(math.pi))
        self.register_buffer('const', torch.tensor(0.5) - self.pi / 4)
        
        # Precompute spline coefficients (will register as buffers)
        self._precompute_spline()
        
    def _precompute_spline(self):
        """Precompute spline coefficients on current device/dtype"""
        device = self.pi.device
        dtype = self.pi.dtype
        
        # Create grid in x-space for middle segment
        x_mid = torch.linspace(
            -self.pi/2, self.pi/2, self.n_mid,
            device=device, dtype=dtype
        )
        
        # Compute corresponding y values using middle segment formula
        y_mid = x_mid - 0.5 * torch.cos(x_mid) + 0.5
        
        # Precompute global constants for linear segments
        self.register_buffer('y_low_global', y_mid[0])     # = 0.5 - pi/2
        self.register_buffer('y_high_global', y_mid[-1])   # = 0.5 + pi/2
        
        # Set up clamped cubic spline system for inverse function
        n = self.n_mid
        A = torch.zeros((n, n), device=device, dtype=dtype)
        rhs = torch.zeros(n, device=device, dtype=dtype)
        
        h_global = torch.diff(y_mid)
        # Left boundary: derivative of inverse = 1/k'(-pi/2) = 1/0.5 = 2.0
        h0 = h_global[0]
        A[0, 0] = 2 * h0
        A[0, 1] = h0
        slope0 = (x_mid[1] - x_mid[0]) / h0
        rhs[0] = 6 * (slope0 - 2.0)
        
        # Right boundary: derivative of inverse = 1/k'(pi/2) = 1/1.5 = 2/3
        h_last = h_global[-1]
        A[-1, -2] = h_last
        A[-1, -1] = 2 * h_last
        slope_last = (x_mid[-1] - x_mid[-2]) / h_last
        rhs[-1] = 6 * ((2.0/3.0) - slope_last)
        
        # Interior points
        for i in range(1, n-1):
            h_prev = h_global[i-1]
            h_curr = h_global[i]
            
            A[i, i-1] = h_prev
            A[i, i] = 2 * (h_prev + h_curr)
            A[i, i+1] = h_curr
            
            slope_forward = (x_mid[i+1] - x_mid[i]) / h_curr
            slope_backward = (x_mid[i] - x_mid[i-1]) / h_prev
            rhs[i] = 6 * (slope_forward - slope_backward)
        
        # Solve for second derivatives
        m_global = torch.linalg.solve(A, rhs)
        
        # Create grouped spline parameters tensor [n_mid-1, 6]
        # Columns: [y_i, x_i, x_{i+1}, m_i, m_{i+1}, h_i]
        n_intervals = self.n_mid - 1
        spline_params = torch.zeros((n_intervals, 6), device=device, dtype=dtype)
        spline_params[:, 0] = y_mid[:-1]          # y_i
        spline_params[:, 1] = x_mid[:-1]          # x_i
        spline_params[:, 2] = x_mid[1:]           # x_{i+1}
        spline_params[:, 3] = m_global[:-1]       # m_i
        spline_params[:, 4] = m_global[1:]        # m_{i+1}
        spline_params[:, 5] = h_global            # h_i
        
        self.register_buffer('spline_params', spline_params)

    def forward(self, x):
        """Compute k(x) for input tensor x"""
        result = torch.zeros_like(x)
        
        # Region 1: x <= -pi/2
        mask1 = x <= -self.pi/2
        result[mask1] = 0.5 * x[mask1] + self.const
        
        # Region 2: -pi/2 < x < pi/2
        mask2 = (x > -self.pi/2) & (x < self.pi/2)
        result[mask2] = x[mask2] - 0.5 * torch.cos(x[mask2]) + 0.5
        
        # Region 3: x >= pi/2
        mask3 = x >= self.pi/2
        result[mask3] = 1.5 * x[mask3] + self.const
        
        return result

    def derivative(self, x):
        """Compute derivative k'(x) = g(x)"""
        result = torch.empty_like(x)
        
        # Region 1: x < -pi/2 -> constant 0.5
        mask1 = x < -self.pi/2
        result[mask1] = 0.5
        
        # Region 2: -pi/2 <= x <= pi/2 -> 0.5*sin(x) + 1
        mask2 = (x >= -self.pi/2) & (x <= self.pi/2)
        result[mask2] = 0.5 * torch.sin(x[mask2]) + 1.0
        
        # Region 3: x > pi/2 -> constant 1.5
        mask3 = x > self.pi/2
        result[mask3] = 1.5
        
        return result

    def inverse(self, y):
        """Compute inverse function k^{-1}(y)"""
        result = torch.empty_like(y)
        
        # Left linear segment: y <= y_low_global
        left_mask = y <= self.y_low_global
        result[left_mask] = 2 * (y[left_mask] - self.const)
        
        # Right linear segment: y >= y_high_global
        right_mask = y >= self.y_high_global
        result[right_mask] = (y[right_mask] - self.const) / 1.5
        
        # Middle segment: spline interpolation
        mid_mask = (~left_mask) & (~right_mask)
        if torch.any(mid_mask):
            y_vals = y[mid_mask]
            
            # Direct index approximation
            normalized = (y_vals - self.y_low_global) / (self.y_high_global - self.y_low_global)
            indices = (normalized * (self.n_mid - 1)).long()
            indices = torch.clamp(indices, 0, self.n_mid - 2)
            
            # Single read of all required parameters
            params = self.spline_params[indices]  # Shape: [num_vals, 6]
            y_i = params[:, 0]
            x_i = params[:, 1]
            x_ip1 = params[:, 2]
            m_i = params[:, 3]
            m_ip1 = params[:, 4]
            h_i = params[:, 5]
            
            # Cubic spline interpolation
            t = (y_vals - y_i) / h_i
            one_minus_t = 1 - t
            h_i2_div6 = (h_i ** 2) / 6
            
            term1 = one_minus_t * x_i + t * x_ip1
            term2 = h_i2_div6 * (one_minus_t ** 3 - one_minus_t) * m_i
            term3 = h_i2_div6 * (t ** 3 - t) * m_ip1
            result[mid_mask] = term1 + term2 + term3
        
        return result
class InvertibleBatchNorm1d(nn.Module):
    """
    Invertible BatchNorm-like layer (feature-wise affine using running stats).
    Always affine: y = ((x - mean) / sqrt(var+eps)) * weight + bias. [web:284]

    forward(x)   -> y   (updates running stats only in train mode)
    inverse(y)   -> x   (never updates running stats)
    derivative(x)-> dy/dx (elementwise diagonal factors; independent of x) [web:284]
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, weight_eps: float = 1e-12):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight_eps = float(weight_eps)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def _reduce_dims(self, x: torch.Tensor):
        # Normalize over all dims except last (features)
        return tuple(range(x.ndim - 1))

    @torch.no_grad()
    def _update_running_stats(self, x: torch.Tensor):
        dims = self._reduce_dims(x)
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)  # typical BN forward variance choice [web:284]

        m = self.momentum
        self.running_mean.mul_(1.0 - m).add_(m * batch_mean)
        self.running_var.mul_(1.0 - m).add_(m * batch_var)
        self.num_batches_tracked.add_(1)

    def forward(self, x: torch.Tensor):
        if self.training:
            self._update_running_stats(x)

        inv_std = torch.rsqrt(self.running_var + self.eps)
        y = (x - self.running_mean) * inv_std
        y = y * self.weight + self.bias
        return y

    def inverse(self, y: torch.Tensor):
        xhat = (y - self.bias) / (self.weight + self.weight_eps)
        std = torch.sqrt(self.running_var + self.eps)
        x = xhat * std + self.running_mean
        return x

    def derivative(self, x: torch.Tensor):
        # dy/dx = weight / sqrt(running_var + eps)
        d = self.weight * torch.rsqrt(self.running_var + self.eps)
        return d.expand_as(x)
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
class InvertibleTanh(nn.Module):
    """
    y = scale * tanh(x)

    Numerical stability improvements:
    - inverse uses torch.atanh when available (stable + simple)
    - fallback uses log1p-based formula
    - clamp uses dtype-aware eps
    - derivative uses cosh formulation to avoid tanh(x)^2 cancellation for large |x|
    """
    def __init__(self, scale=2.0, eps=None):
        super().__init__()
        self.scale = float(scale)
        self.eps = eps  # if None -> choose from dtype at runtime

    def _eps(self, x):
        # dtype-aware epsilon; you can also force a larger eps via ctor
        if self.eps is not None:
            return float(self.eps)
        # tiny is the smallest positive normal; eps is machine epsilon
        # Using a slightly larger clamp than machine eps is usually better here.
        return 1e-6 if x.dtype in (torch.float32, torch.bfloat16) else 1e-12

    def forward(self, x):
        return self.scale * torch.tanh(x)

    def inverse(self, y):
        # x = atanh(y/scale), but must keep |y/scale| < 1
        z = y / self.scale
        eps = self._eps(z)
        z = z.clamp(min=-1.0 + eps, max=1.0 - eps)

        # Prefer builtin atanh (stable implementation)
        if hasattr(torch, "atanh"):
            return torch.atanh(z)

        # Fallback: atanh(z) = 0.5 * (log1p(z) - log1p(-z))  (more stable than log((1+z)/(1-z)))
        return 0.5 * (torch.log1p(z) - torch.log1p(-z))

    def derivative(self, x):
        # dy/dx = scale * sech^2(x) = scale / cosh(x)^2
        # This avoids tanh(x)^2 cancellation when tanh(x) saturates near 1.
        c = torch.cosh(x)
        return self.scale / (c * c)
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
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _sqrt1pm1_stable(t: torch.Tensor, eps: float):
        # returns sqrt(1+t) - 1, stable for small t
        # valid for t >= -1
        s = torch.sqrt(torch.clamp(1.0 + t, min=0.0))
        return t / (s + 1.0 + eps)  # use identity from stable sqrt difference [web:210]

    def forward(self, x: torch.Tensor):
        # y = sign(x) * (sqrt(1+|x|) - 1)
        xa = x.abs()
        y = self._sqrt1pm1_stable(xa, self.eps)
        return torch.copysign(y, x)

    def inverse(self, y: torch.Tensor):
        # if y =  sqrt(1+x)-1  => x = (y+1)^2 - 1
        # if y = - (sqrt(1+|x|)-1) => |x| = (|y|+1)^2 - 1, then apply sign
        ya = y.abs()
        xa = (ya + 1.0).pow(2) - 1.0
        return torch.copysign(xa, y)

    def derivative(self, x: torch.Tensor):
        # d/dx [sign(x)*(sqrt(1+|x|)-1)] = 1/(2*sqrt(1+|x|)) for x!=0
        xa = x.abs()
        denom = 2.0 * torch.sqrt(1.0 + xa)  # already stable; no subtraction here
        # avoid divide by 0 if something goes pathological
        return 1.0 / (denom + self.eps)
class SmoothSymmetricSqrt(nn.Module):
    """
    This function is kinda-interpolation between y=x and y=sqrt(x)
    """
    def __init__(self,inv_split = 2.5):
        super().__init__()
        
        self.inv_split = inv_split
        b = [0.0038933507013838354,
            1.0526701833814136,
            -1.3141444069161157,
            0.6846405296018866,
            -0.08648287900443206,
            -1.3859825762259927,
            0.823326685232045,
            -0.20215808335524396,
            0.018149838932938412,
            0.9045875826093197]
        inv_weights=torch.tensor(b)
        self.register_buffer('inv_weights', inv_weights)
        self.A = 1.66
        self.B = -0.31630601136
        self.bias = -0.8431526
    
    def forward(self, x):
        sign = x.sign()
        x=x.abs()        
        a = 2*(x+1).sqrt()
        b = x
        sigmoid = (self.A*x+self.B).sigmoid()
        return sign*0.555*(a*sigmoid+(1-sigmoid)*b+self.bias)
    
    def derivative(self, x):
        y = x.abs()
        s = (self.A * y + self.B).sigmoid()
        sqrt_term = (y + 1).sqrt()
        
        # Compute h'(y) = A*s*(1-s)*(sqrt(y+1) - 0.5*y) + s/(2*sqrt(y+1)) + 0.5*(1-s)
        term1 = self.A * s * (1 - s) * (sqrt_term - 0.5 * y)
        term2 = s / (2 * sqrt_term)
        term3 = 0.5 * (1 - s)
        
        return (term1 + term2 + term3)*1.11
    
    def inverse(self,x : torch.Tensor):
        """This is somewhat optimal approximation of Smooth Symmetric Sqrt inverse function"""
        x=x/1.11
        sign = x.sign()
        x = x.abs()
        res = torch.zeros_like(x)
        
        x1_mask = x>=2.5792
        if torch.any(x1_mask):
            x1 = x[x1_mask]
            inv = (x1-self.bias/2).pow(2)-1
            res[x1_mask]=inv*sign[x1_mask]
        
        x2_mask = x<=0.05
        if torch.any(x2_mask):
            x2 = x[x2_mask]
            res[x2_mask]=x2*sign[x2_mask]/self.inv_weights[-1]
        
        x0_mask = ~x1_mask & ~x2_mask
        if torch.any(x0_mask):
            w=self.inv_weights
            x0 = x[x0_mask]
            x1 = x0
            x2 = x0*x1
            x3 = x0*x2
            x4 = x0*x3
            up = w[0]+x1*w[1]+x2*w[2]+x3*w[3]+x4*w[4]
            down = 1+w[5]*x1+w[6]*x2+w[7]*x3+w[8]*x4
            res[x0_mask] = up/down*sign[x0_mask]
            
        return res
class InvertibleIdentity(nn.Module):
    """
    Invertible no-op / placeholder nonlinearity.

    - forward(x) returns x unchanged.
    - inverse(y) returns y unchanged.
    - derivative(x) returns ones with same shape as x (Jacobian determinant = 1 elementwise).

    This is analogous to torch.nn.Identity, which is a placeholder identity operator. [web:304]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

def create_checkerboard_mask(dims,device):
    """
    Create checkerboard mask for spatial partitioning
    Returns mask where True values remain unchanged, False values get transformed
    """
    mask = torch.zeros(dims, dtype=torch.bool,device=device)
    a = [torch.arange(d,device=device) for d in dims]
    ab = torch.meshgrid(a,indexing="ij")
    mask[torch.stack(ab).sum(0)%2==0]=True
    return mask

def split_tensor(x,split_dim):
    ind = create_checkerboard_mask(x.shape[1:],device=x.device)
    reduced_shape = list(x.shape)
    reduced_shape[split_dim]//=2
    set_a = x[:,ind].reshape(reduced_shape)
    set_b = x[:,~ind].reshape(reduced_shape)   
    return set_a,set_b 

def join_split(a,b,split_dim):
    shape = list(a.shape)
    shape[split_dim]*=2
    res = torch.zeros(shape,device=a.device)
    ind = create_checkerboard_mask(res.shape[1:],a.device)
    res[:,ind]=a.reshape(a.shape[0],-1)
    res[:,~ind]=b.reshape(b.shape[0],-1)
    return res

def permute_even_odd(x: torch.Tensor, dim: int = -1):
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

def unpermute_even_odd(x: torch.Tensor, dim: int = -1):
    """
    Undo permute_even_odd: interleave even and odd parts back along a given dim.

    Args:
        x: tensor produced by permute_even_odd
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
        flip_dim: keep it None please
    """
    def __init__(
        self, 
        model,
        dimension_split = -1,
        non_linearity : Union[InvertibleTanh,SmoothSymmetricSqrt,InvertibleIdentity] = SmoothSymmetricSqrt,
    ):
        super().__init__()
        self.model=model
        self.dimension_split = dimension_split  # Ensure integer type
        if isinstance(non_linearity,type):
            non_linearity=non_linearity()
        self.non_linearity=non_linearity
        self.flip = random.randint(0,1)==0
    
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
        y=self.non_linearity(z2)
        
        concat = torch.concat([y,x1],self.dimension_split)
        concat = permute_even_odd(concat,self.dimension_split)
        
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
        output = unpermute_even_odd(output,self.dimension_split)
        f_z2,x1 = output.chunk(2,self.dimension_split)
        
        z2 = self.non_linearity.inverse(f_z2)
        
        scale,translate = self.get_scale_and_translate(x1)
        x2 = (z2-translate)/(scale+1e-6)
        
        # concat = join_split(x1,x2,self.dimension_split)
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

def flow_nll_loss(z, jacobians, x, eps: float = 1e-8,sum_dim=-1):
    """
    Maximum-likelihood loss for a normalizing flow.

    Args:
        z: Latents of normalizing flow models
        jacobians: jacobians of normalizing flow model
        x: Data batch shaped (B, ...).
        eps: Numerical stability for log.
        sum_dim: Which dimension to reduce over *after* flattening non-batch dims with
            `.flatten(1)`. In this implementation, both `log_det` and `log_pz` are built as
            tensors of shape (B, N) where N is the number of event dimensions (all original
            dims except batch). `sum_dim` controls which dimension you sum across to get one
            scalar log-probability per sample. Typically keep `sum_dim=-1` (or `sum_dim=1`)
            so you sum over the event dimension(s) and end up with shape (B,) for `log_det`
            and `log_pz`. [web:259]

            Examples:
            - If `flatten(1)` produces shape (B, N), then `sum_dim=1` or `sum_dim=-1` gives
              per-sample totals.
            - If you later change the code to not flatten, you could use a tuple of dims
              (but this function currently expects an int because it flattens first).

    Returns:
        loss: Scalar (mean NLL over batch).
        diagnostics: dict with mean log_det and mean log_pz.
    """

    # 1) log|det J| for the full flow: sum over layers, sum over event dims
    log_det = 0.0
    for jd in jacobians:
        safe_abs1 = jd.abs() + eps
        # safe_abs2 = torch.nn.functional.smooth_l1_loss(jd,torch.zeros_like(jd),reduction='none')+eps
        # avg = (safe_abs1+safe_abs2)/2
        # jd shape matches the transformed subset; sum over all non-batch dims
        log_det += torch.log(safe_abs1).flatten(1)

    # 2) log p(z) under N(0,1): sum over event dims
    # log N(z;0,1) = -0.5*(z^2 + log(2*pi)) per dimension
    log_pz = (-0.5 * (z**2 + math.log(2 * math.pi))).flatten(1)
    
    if sum_dim is not None:
        log_det=log_det.sum(dim=sum_dim)
        log_pz=log_pz.sum(dim=sum_dim)

    # 3) log p(x) = log p(z) + log|det J|
    log_px = log_pz + log_det

    return -log_px,log_det
