from typing import Tuple
from kemsekov_torch.attention import SelfAttention, EfficientSpatialChannelAttention, zero_module
from kemsekov_torch.residual import Residual,ResidualBlock
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self,in_channels,emb_dim = None,latent_dim = None) -> None:
        super().__init__()
        if emb_dim is None:
            emb_dim=in_channels*16
        if latent_dim is None:
            latent_dim=emb_dim
        self.in_channels = in_channels
        self.latent_dim=latent_dim
        self.encoder = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(in_channels*16,emb_dim,1),
            
            Residual([
                SelfAttention(emb_dim,add_absolute_pos=False,add_rotary_embedding=True,abs_pos_jit_prob=0),
                nn.GroupNorm(16,emb_dim),
                EfficientSpatialChannelAttention(emb_dim),
                nn.SiLU(),
                zero_module(nn.Conv2d(emb_dim,emb_dim,1)),
            ],init_at_zero=False),
            
            Residual([
                SelfAttention(emb_dim,add_absolute_pos=False,add_rotary_embedding=True,abs_pos_jit_prob=0),
                nn.GroupNorm(16,emb_dim),
                EfficientSpatialChannelAttention(emb_dim),
                nn.SiLU(),
                zero_module(nn.Conv2d(emb_dim,emb_dim,1)),
            ],init_at_zero=False),
            
            nn.Conv2d(emb_dim,latent_dim*2,1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim,emb_dim,1),
            
            Residual([
                SelfAttention(emb_dim,add_absolute_pos=False,add_rotary_embedding=True,abs_pos_jit_prob=0,prenorm='group'),
                nn.GroupNorm(16,emb_dim),
                EfficientSpatialChannelAttention(emb_dim),
                nn.SiLU(),
                zero_module(nn.Conv2d(emb_dim,emb_dim,1)),
            ],init_at_zero=False),
            
            Residual([
                SelfAttention(emb_dim,add_absolute_pos=False,add_rotary_embedding=True,abs_pos_jit_prob=0,prenorm='group'),
                nn.GroupNorm(16,emb_dim),
                EfficientSpatialChannelAttention(emb_dim),
                nn.SiLU(),
                zero_module(nn.Conv2d(emb_dim,emb_dim,1)),
            ],init_at_zero=False),
            
            nn.Conv2d(emb_dim,in_channels*16,1),
            nn.PixelShuffle(4),
            
            # nn.Conv2d(emb_dim,in_channels,1)
        )
    
    def encode(self,x : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        mu,logstd = self.encoder(x).chunk(2,1)
        return mu,logstd
    
    def decode(self,z : torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def sample(self,mu : torch.Tensor,logstd : torch.Tensor) -> torch.Tensor:
        return torch.randn_like(mu)*logstd.exp()+mu
    
    def forward(self,x : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        mu,logstd = self.encode(x)
        z = self.sample(mu,logstd)
        dec = self.decode(z)
        return mu,logstd,z,dec
    
# vae = VAE(3,emb_dim=512,latent_dim=6)