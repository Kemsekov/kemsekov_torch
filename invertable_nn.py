import torch
import torch.nn as nn

class InvertableScaleAndTranslate(nn.Module):
    """
    Invertible neural network for normalizing flows, applying scaling, translation, and shuffling.
    
    Args:
        scale_model (nn.Module): Neural network to compute scaling factors.
        translate_model (nn.Module): Neural network to compute translation factors.
        dimension_split (int, optional): Dimension to split the input. Defaults to -1 (last dimension).
        seed (int, optional): Seed for reproducible shuffling. If None, a random integer is generated.
    """
    def __init__(self, scale_model,translate_model,dimension_split = -1,seed = None):
        super().__init__()
        self.scale_model=scale_model
        self.translate_model=translate_model
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
            torch.Tensor: Transformed tensor with the same shape as input.
        """
        x1,x2 = input.chunk(2,self.dimension_split)
        scale = self.scale_model(x1)
        translate = self.translate_model(x1)
        z2 = x2*scale+translate

        concat = torch.concat([x1,z2],self.dimension_split)
        generator = torch.Generator().manual_seed(self.seed)
        shuffle_ind = torch.rand(concat.shape[self.dimension_split],generator=generator).argsort()
        concat_shuffle = concat.index_select(self.dimension_split,shuffle_ind)
        
        return concat_shuffle

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
        scale = self.scale_model(x1)+1e-6
        translate = self.translate_model(x1)
        x2 = (z2-translate)/scale
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
        
