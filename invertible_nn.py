import random
from typing import Optional, Union
import torch
import torch.nn as nn
import math
from copy import deepcopy

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
        return sign*0.5*(a*sigmoid+(1-sigmoid)*b+self.bias)
    
    def derivative(self, x):
        y = x.abs()
        s = (self.A * y + self.B).sigmoid()
        sqrt_term = (y + 1).sqrt()
        
        # Compute h'(y) = A*s*(1-s)*(sqrt(y+1) - 0.5*y) + s/(2*sqrt(y+1)) + 0.5*(1-s)
        term1 = self.A * s * (1 - s) * (sqrt_term - 0.5 * y)
        term2 = s / (2 * sqrt_term)
        term3 = 0.5 * (1 - s)
        
        return term1 + term2 + term3
    
    def inverse(self,x : torch.Tensor):
        """This is somewhat optimal approximation of Smooth Symmetric Sqrt inverse function"""
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

def flow_nll_loss(flow, x, eps: float = 1e-12,sum_dim=-1):
    """
    Maximum-likelihood loss for a normalizing flow.

    Args:
        flow: InvertibleSequential. forward(x) -> (z, jacobians)
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
    z, jacobians = flow(x)  # z = f(x)

    # 1) log|det J| for the full flow: sum over layers, sum over event dims
    log_det = 0.0
    for jd in jacobians:
        safe_abs = jd.abs() + eps
        # smooth_l1 = torch.nn.functional.smooth_l1_loss(jd,torch.zeros_like(jd),reduction='none')+1e-2
        # jd shape matches the transformed subset; sum over all non-batch dims
        log_det += torch.log(safe_abs).flatten(1).sum(dim=sum_dim)

    # 2) log p(z) under N(0,1): sum over event dims
    # log N(z;0,1) = -0.5*(z^2 + log(2*pi)) per dimension
    log_pz = (-0.5 * (z**2 + math.log(2 * math.pi))).flatten(1).sum(dim=sum_dim)

    # 3) log p(x) = log p(z) + log|det J|
    log_px = log_pz + log_det

    # maximize log p(x)  <=>  minimize -log p(x)
    nll = -log_px
    loss = nll.mean()

    return loss
from torch.distributions import Normal
from kemsekov_torch.residual import Residual
class NormalizingFlow:
    """
    Wrapper around your InvertibleSequential + flow_nll_loss training loop.

    Key features:
    - Model definition is fully determined in __init__ (input_dim is required, not inferred from data).
    - fit(...) trains on a tensor dataset and returns the best model (CPU, eval).
    - Works with flow_nll_loss that returns either:
        * loss
        * (loss, diagnostics_dict)
      (avoids "iteration over a 0-d tensor" unpacking error).
    - Optional gradient clipping via torch.nn.utils.clip_grad_norm_. [web:381]
    - Uses optimizer.zero_grad(set_to_none=True) for performance/memory. [web:399]
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 32,
        layers: int = 3,
        device: Optional[str] = 'cpu',
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self._build_model().to(self.device)
        self.best_trained_model = None

    def to(self,device):
        self.device=device
        self.model=self.model.to(device)
        if self.best_trained_model:
            self.best_trained_model=self.best_trained_model.to(device)

    def _build_model(self) -> nn.Module:
        if self.input_dim % 2 != 0:
            raise ValueError(
                f"input_dim must be even for InvertibleScaleAndTranslate(input.chunk(2)). Got {self.input_dim}."
            )

        norm = nn.RMSNorm
        act = nn.ReLU
        # act = nn.SiLU
        
        half = self.input_dim // 2
        blocks = []
        for i in range(self.layers):
            steps = [
                nn.Linear(half, self.hidden_dim),
                
                Residual([
                    norm(self.hidden_dim),
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ],init_at_zero=True),
                
                Residual([
                    norm(self.hidden_dim),
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ],init_at_zero=True),
                
                norm(self.hidden_dim),
                act(),

                nn.Linear(self.hidden_dim, self.input_dim),
                # norm(self.input_dim)
            ]
            if i==self.layers-1 and "Norm" in str(steps[-1]):
                steps=steps[:-1]
            blocks.append(
                InvertibleScaleAndTranslate(
                    model=nn.Sequential(*steps),
                    dimension_split=-1,
                    non_linearity=InvertibleIdentity
                    # non_linearity=InvertibleTanh
                    # non_linearity=SmoothSymmetricSqrt
                    # non_linearity=InvertibleLeakyReLU
                )
            )
        blocks[-1].non_linearity = InvertibleIdentity()
        return InvertibleSequential(*blocks)
    # TODO: add sample method
    # add optimize method
    def log_prob(self, data):
        model = self.best_trained_model or self.model
        z, jacobians = model(data.to(self.device))
        
        # log p(z) under standard normal
        log_pz = Normal(0, 1).log_prob(z).flatten(1).sum(dim=-1)
        
        # log |det J|
        log_det = 0.0
        for jd in jacobians:
            log_abs_jd = torch.log(torch.abs(jd) + 1e-12)
            log_det += log_abs_jd.flatten(1).sum(dim=-1)
        
        # log p(x) = log p(z) + log |det J|
        log_px = log_pz + log_det
        
        return log_px.to(data.device)
    def fit(
        self,
        data: torch.Tensor,
        batch_size: int = 512,
        epochs: int = 30,
        lr: float = 1e-2,
        data_renoise=0.05,
        grad_clip_max_norm: Optional[float] = None,
        debug: bool = True,
    ) -> nn.Module:
        """
        Train on `data` and return best model.

        Args:
            data: Tensor of shape [N, input_dim].
            batch_size: Batch size.
            lr: AdamW learning rate.
            epochs: Epoch count.
            save_skip_epochs: Start tracking best model after this epoch index.
            grad_clip_max_norm: If not None, clip global grad norm to this value. [web:381]
            debug: If True, prints when best loss improves.

        Returns:
            trained_model: Best model on CPU in eval() mode.
        """
        if data.ndim != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Expected data shape [N, {self.input_dim}], got {tuple(data.shape)}")

        batch_size = min(batch_size,data.shape[0])
        data = data.to(self.device)
        
        data_min_std = data.std(0).quantile(0)

        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        best_loss = float("inf")
        self.best_trained_model = deepcopy(self.model).to(self.device)
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,len(slices)*epochs)
        try:
            for epoch in range(epochs):
                if debug and improved:
                    print(f"Epoch {epoch}: best_loss={best_loss:0.3f}")
                improved = False

                # shuffle each epoch
                perm = torch.randperm(n, device=self.device)
                data_shuf = data[perm]

                for start in slices:
                    batch = data_shuf[start : start + batch_size]
                    
                    if data_renoise>0:
                        batch=batch+torch.randn_like(batch)*data_min_std*data_renoise
                    
                    optim.zero_grad(set_to_none=True)  # set_to_none saves mem and can be faster [web:399]
                    loss = flow_nll_loss(self.model, batch, sum_dim=-1)

                    if loss < best_loss:
                        best_loss = loss
                        self.best_trained_model = deepcopy(self.model).to(self.device)
                        improved = True
                    loss.backward()
                    if grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm,
                            norm_type=2.0,
                        )
                    optim.step()
                    sch.step()
                    
        except KeyboardInterrupt:
            if debug: print("Stop training")
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}")
        return self.best_trained_model.eval()
    