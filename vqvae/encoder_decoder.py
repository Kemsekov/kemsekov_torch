from typing import List
import torch
from kemsekov_torch.residual import ResidualBlock
import torch.nn as nn
# Three layers hierarchical encoder and decoder

class Encoder(nn.Module):
    def __init__(self,input_channels,h_dim,internal_dim=128,num_residual_layers=2,normalization = 'batch',dimensions=2):
        """
        input_channels: input channels
        h_dim: output channels dimension
        internal_dim: internal dimension of encoder, in general it is greater than `h_dim`
        num_residual_layers: number of residual layers
        normalization: one of `['batch','instance','group',None]` normalization that is used in convolutions
        dimensions: one of `[1,2,3]` input dimension of encoder
        """
        super().__init__()
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        common={
            'normalization':normalization,
            "dimensions":dimensions
        }
        
        res_dim=internal_dim//4
        self.enc = nn.Sequential(
            # nn.Dropout(0.1), #denoise autoencoder
            ResidualBlock(input_channels,internal_dim//2,stride=2,kernel_size=4,**common),
            ResidualBlock(internal_dim//2,internal_dim,stride=2,kernel_size=4,**common),
            
            # residual stack
            *[
                ResidualBlock(internal_dim,[res_dim,internal_dim],**common) 
                for i in range(num_residual_layers)
            ],
            ResidualBlock(internal_dim,h_dim,**common,kernel_size=1) 
        )
        self.high = conv(h_dim,h_dim,kernel_size=1)
        self.middle = ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,**common)
        self.middle_out = conv(h_dim,h_dim,kernel_size=1)
        self.bottom = nn.Sequential(
            ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,**common),
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
    def __init__(self,output_channels,h_dim,internal_dim=128,num_residual_layers=2,normalization = 'batch',dimensions=2):
        """
        output_channels: output channels
        h_dim: output channels dimension
        internal_dim: internal dimension of encoder, in general it is greater than `h_dim`
        num_residual_layers: number of residual layers
        normalization: one of `['batch','instance','group',None]` normalization that is used in convolutions
        dimensions: one of `[1,2,3]` input dimension of encoder
        """
        super().__init__()
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        common={
            'normalization':normalization,
            "dimensions":dimensions
        }
        self.high_conv = ResidualBlock(h_dim,h_dim,**common)
        self.middle_upscale = ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,**common).transpose()
        self.low_upscale = nn.Sequential(
            ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,**common).transpose(),
            ResidualBlock(h_dim,h_dim,kernel_size=4,stride=2,**common).transpose()
        )
        
        res_dim=internal_dim//4
        self.dec = nn.Sequential(
            ResidualBlock(3*h_dim,internal_dim,kernel_size=1,**common),
            *[
                ResidualBlock(internal_dim,[res_dim,internal_dim],**common) 
                for i in range(num_residual_layers)
            ],
            ResidualBlock(internal_dim,internal_dim//2,stride=2,kernel_size=4,**common).transpose(),
            ResidualBlock(internal_dim//2,h_dim,stride=2,kernel_size=4,**common).transpose(),
            conv(h_dim,output_channels,kernel_size=1)
        )

    def forward(self,x : List[torch.Tensor]):
     x_high,x_middle,x_low = x
     x_high = self.high_conv(x_high)
     x_middle=self.middle_upscale(x_middle)
     x_low=self.low_upscale(x_low)
     x = torch.concat([x_high,x_middle,x_low],dim=1)
     return self.dec(x)
