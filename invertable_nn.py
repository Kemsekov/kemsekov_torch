import torch
import torch.nn as nn

class InvertableLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope = negative_slope  # Store negative_slope for consistency

    def forward(self, x):
        # Compute Leaky ReLU output
        output = self.leaky_relu(x)
        # Compute derivative: 1 if x >= 0, negative_slope if x < 0
        derivative = torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, self.negative_slope))
        return output, derivative

    def inverse(self, x):
        # Compute inverse: y if y >= 0, y/negative_slope if y < 0
        output = torch.where(x >= 0, x, x / self.negative_slope)
        return output

class InvertableTanh(torch.nn.Module):
    def __init__(self,scale=1):
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

class SymetricLog(nn.Module):
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

class SymetricSqrt(nn.Module):
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
    
class InvertableScaleAndTranslate(nn.Module):
    """
    Invertible neural network for normalizing flows, applying scaling, translation, and shuffling with nonlinear function, which provides infinitely differentiable invertable neural network.
    
    Args:
        model (nn.Module): Neural network to compute scaling and translation factors. It must returns twise dimensions as it takes as input.
        dimension_split (int, optional): Dimension to split the input. Defaults to -1 (last dimension).
        seed (int, optional): Seed for reproducible shuffling. If None, a random integer is generated.
    """
    def __init__(self, model,dimension_split = -1,seed = None):
        super().__init__()
        self.model=model
        if seed is None:
            seed = torch.randint(0, 1000, [1])[0].item()  # Generate random integer
        self.seed = seed
        self.dimension_split = dimension_split  # Ensure integer type
        self.non_linearity = InvertableTanh(2)
    
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
        f_z2,x1 = output.chunk(2,self.dimension_split)
        z2 = self.non_linearity.inverse(f_z2)
        
        scale,translate = self.get_scale_and_translate(x1)
        x2 = (z2-translate)/(scale+1e-6)
        concat = torch.concat([x1,x2],self.dimension_split)
        return concat
  
class InvertableSequential(nn.Sequential):
    """
    Sequential container for invertible modules, supporting forward and inverse transformations.
    
    Args:
        *modules: Sequence of invertible nn.Module instances.
    """
    def __init__(self, *modules):
        super().__init__(*modules)
        
    def forward(self,input):
        """
        Computes forward pass of invertable neural networks composition.
        
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
        
