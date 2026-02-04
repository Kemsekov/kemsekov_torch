import torch
import torch.nn as nn
import torch.nn.functional as F
from kemsekov_torch.flow_matching import FlowMatching
from kemsekov_torch.residual import Residual
from kemsekov_torch.attention import EfficientSpatialChannelAttention
import torch.nn as nn
from kemsekov_torch.attention import SelfAttention, EfficientSpatialChannelAttention
from kemsekov_torch.residual import Residual

class ViT(nn.Module):
    def __init__(
            self, 
            in_channels,
            hidden_dim=64,
            expand_dim=256,
            layers = 3,
            heads=8,
            head_dim=64,
            compression = 4
        ) -> None:
        super().__init__()
        
        conv = nn.Conv2d
        self.compression=compression
        self.down = nn.Sequential(
            nn.PixelUnshuffle(compression),
            conv(in_channels*(compression**2),hidden_dim,1),
        )
        
        groups = max(1,hidden_dim//32)
        if groups==1: groups=2
        
        # my self and cross attention works on conv2d-shaped tensors
        # [B,C,H,W] where H,W is actual tokens
        
        self.residuals = nn.ModuleList(
            [
                nn.ModuleList([
                    Residual([
                        SelfAttention(hidden_dim,heads=heads,head_dim=head_dim,add_rotary_embedding=True,dropout=0.0),
                        nn.GroupNorm(groups,hidden_dim),
                        nn.SiLU(),
                        Residual([
                            conv(hidden_dim,expand_dim,1),
                            nn.GroupNorm(32,expand_dim),
                            EfficientSpatialChannelAttention(expand_dim,kernel_size=3),
                            nn.SiLU(),
                            conv(expand_dim,hidden_dim,1),
                        ])
                    ],init_at_zero=False),
                    nn.Sequential(
                        nn.Linear(1,hidden_dim),
                        # nn.RMSNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim,hidden_dim*2),
                    ),
                ])
                for i in range(layers)
            ]
        )
        
        self.up = nn.Sequential(
            nn.GroupNorm(groups,hidden_dim),
            conv(hidden_dim,in_channels*(compression**2),1),
            nn.PixelShuffle(compression),
        )
        self.pos_gamma = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self,x,t):
        # x_orig = x
        while t.ndim==1:
            t = t.unsqueeze(-1)
        
        x = self.down(x)
        for r,time_emb in self.residuals:
            time_scale,time_shift = time_emb(t)[:,:,None,None].chunk(2,1)
            xt = x*(1+time_scale)+time_shift
            x = r(xt)
        return self.up(x)

class FlowModel2d(nn.Module):
    def __init__(self, in_channels,hidden_dim = 256) -> None:
        super().__init__()
        self.default_steps = 16
        self.fm = FlowMatching()
        # self.fm.time_scaler = lambda x: torch.log(9*x+1)/math.log(10)
        self.in_channels=in_channels
        self.vit = ViT(
            in_channels,
            hidden_dim=hidden_dim,
            expand_dim=hidden_dim*4,
            layers=10,
            head_dim=64,
            compression=4
        )
        # self.ln = LossNormalizer2d(in_channels,64)
        self.device='cpu'
    
    def forward(self,x,t):
        return self.vit(x,t)
    
    def to(self,device):
        self.device = device
        return super().to(device)
    
    @torch.compiler.disable
    def to_prior(self,data : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        input_device = data.device
        return self.fm.integrate(self,data.to(self.device),steps,inverse=True).to(input_device)
    
    @torch.compiler.disable
    def to_target(self,normal_noise : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        input_device = normal_noise.device
        return self.fm.integrate(self,normal_noise.to(self.device),steps).to(input_device)

    @torch.compiler.disable
    def sample(self,num_samples,image_size=(64,64),steps=None):
        if not steps: steps = self.default_steps
        return self.to_target(torch.randn((num_samples,self.in_channels,*image_size),device=self.device),steps)