from typing import List
import torch
from kemsekov_torch.residual import ResidualBlock
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,h_dim,internal_dim,res_h_dim,num_residual_layers=2,normalization = 'batch',dimensions=2):
        super().__init__()
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        self.enc = nn.Sequential(
            # nn.Dropout(0.1), #denoise autoencoder
            ResidualBlock(3,h_dim//2,stride=2,kernel_size=4,normalization=normalization,dimensions=dimensions),
            ResidualBlock(h_dim//2,internal_dim,stride=2,kernel_size=4,normalization=normalization,dimensions=dimensions),
            
            # residual stack
            *[
                ResidualBlock(internal_dim,[res_h_dim,internal_dim],normalization=normalization,dimensions=dimensions) 
                for i in range(num_residual_layers)
            ],
        )
        self.high = conv(internal_dim,h_dim,kernel_size=1)
        self.middle = ResidualBlock(internal_dim,internal_dim,kernel_size=4,stride=2,normalization=normalization,dimensions=dimensions)
        self.middle_out = conv(internal_dim,h_dim,kernel_size=1)
        self.bottom = nn.Sequential(
            ResidualBlock(internal_dim,h_dim,kernel_size=4,stride=2,dimensions=dimensions),
            conv(h_dim,h_dim,kernel_size=1)
        )
        
    def forward(self,x):
        x = self.enc(x)
        h = self.high(x)
        m = self.middle(x)
        b = self.bottom(m)
        m = self.middle_out(m)
        
        return [h,m,b]
class Decoder(nn.Module):
    def __init__(self,h_dim,internal_dim,res_h_dim,num_residual_layers=2,normalization = 'batch',dimensions=2):
        super().__init__()
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        self.high_conv = ResidualBlock(h_dim,h_dim,normalization=None,dimensions=dimensions)
        self.middle_upscale = ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,normalization=None,dimensions=dimensions).transpose()
        self.low_upscale = nn.Sequential(
            ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,normalization=None,dimensions=dimensions).transpose(),
            ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,normalization=normalization,dimensions=dimensions).transpose()
        )
        
        self.dec = nn.Sequential(
            ResidualBlock(3*h_dim,internal_dim,kernel_size=1,normalization=normalization,dimensions=dimensions),
            
            # because of quantization in the bottleneck, running_mean at first conv will become NAN, 
            # so we do not use normalization here
            *[
                ResidualBlock(internal_dim,[res_h_dim,internal_dim],normalization=normalization,dimensions=dimensions) 
                for i in range(num_residual_layers)
            ],
            ResidualBlock(internal_dim,h_dim//2,stride=2,kernel_size=4,normalization=normalization,dimensions=dimensions).transpose(),
            ResidualBlock(h_dim//2,h_dim//4,stride=2,kernel_size=4,normalization=normalization,dimensions=dimensions).transpose(),
            conv(h_dim//4,3,kernel_size=1)
        )

    def forward(self,x : List[torch.Tensor]):
     x_high,x_middle,x_low = x
     x_high = self.high_conv(x_high)
     x_middle=self.middle_upscale(x_middle)
     x_low=self.low_upscale(x_low)
     x = torch.concat([x_high,x_middle,x_low],dim=1)
     return self.dec(x)
