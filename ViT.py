import torch.nn as nn
from kemsekov_torch.attention import SelfAttention, EfficientSpatialChannelAttention
from kemsekov_torch.residual import Residual

class ViT(nn.Module):
    def __init__(
            self, 
            in_channels,
            hidden_dim=64,
            layers = 3,
            heads=8,
            head_dim=64,
        ) -> None:
        super().__init__()
        
        compression = 4
        conv = nn.Conv2d
        
        self.down = nn.Sequential(
            nn.PixelUnshuffle(compression),
            conv(in_channels*(compression**2),hidden_dim,1),
        )
        
        groups = max(1,hidden_dim//32)
        if groups==1: groups=2
        
        self.residuals = nn.ModuleList(
            [
                Residual([
                    SelfAttention(hidden_dim,heads=heads,head_dim=head_dim,add_rotary_embedding=True),
                    nn.GroupNorm(groups,hidden_dim),
                    EfficientSpatialChannelAttention(hidden_dim),
                    nn.SiLU(),
                ],init_at_zero=False)
                for i in range(layers)
            ]
        )
        
        self.up = nn.Sequential(
            nn.GroupNorm(groups,hidden_dim),
            conv(hidden_dim,in_channels*(compression**2),1),
            nn.PixelShuffle(compression),
        )
        
    def forward(self,x):
        x = self.down(x)
        for r in self.residuals:
            x = r(x)
        return self.up(x)
