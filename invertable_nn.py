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

class InvertableScaleAndTranslate(nn.Module):
    """
    Invertible neural network for normalizing flows, applying scaling, translation, and shuffling.
    
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
        scale,translate = self.model(x1).chunk(2,self.dimension_split)
        z2 = x2*scale+translate
        concat = torch.concat([x1,z2],self.dimension_split)
        concat_deriv = torch.concat([torch.ones_like(x1),scale],self.dimension_split)
        
        generator = torch.Generator().manual_seed(self.seed)
        shuffle_ind = torch.rand(concat.shape[self.dimension_split],generator=generator).argsort()
        concat_shuffle = concat.index_select(self.dimension_split,shuffle_ind)
        concat_shuffle_deriv = concat_deriv.index_select(self.dimension_split,shuffle_ind)
        
        return concat_shuffle, concat_shuffle_deriv

    def inverse(self,output):
        """
        Inverse transformation: deshuffle, split, invert scale/translate, and concatenate.
        
        Args:
            output (torch.Tensor): Output tensor from forward pass.
        
        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        generator = torch.Generator().manual_seed(self.seed)
        shuffle_ind = torch.rand(output.shape[self.dimension_split],generator=generator).argsort().argsort()
        output = output.index_select(self.dimension_split,shuffle_ind)
        
        x1,z2 = output.chunk(2,self.dimension_split)
        scale,translate = self.model(x1).chunk(2,self.dimension_split)
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
        output and list of jacobians
        """
        out = input
        det_jacobian = torch.ones_like(input)
        for m in self:
            out,deriv = m(out)
            det_jacobian*=deriv
        
        return out,det_jacobian
    
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
        
