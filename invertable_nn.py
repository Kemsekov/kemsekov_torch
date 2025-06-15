import torch
import torch.nn as nn

class InvertableScaleAndTranslate(nn.Module):
    """
    Invertable neural network.
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
    def __init__(self, *modules):
        super().__init__(*modules)
    def inverse(self,out):
        prev = out
        for m in reversed(self):
            prev = m.inverse(prev)
        return prev
        
