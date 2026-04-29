from kemsekov_torch.common_modules import resize_tensor, Prod
from kemsekov_torch.attention import SelfAttention
import torch.nn as nn

def sum_tensors(a,b):
    if a.shape!=b.shape:
        b = resize_tensor(b,a.shape[1:])
    return a+b

class Unet(nn.Module):
    def __init__(self, in_channels,out_channels,compression = 4,layers_scaler=1,dropout=0):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        
        layer_dims = [32,64,128,256]
        for i in range(4): layer_dims[i]=int(layer_dims[i]*layers_scaler)
        
        self.compress = nn.Sequential(
            nn.PixelUnshuffle(compression),
            nn.Conv2d(compression**2*in_channels,layer_dims[0],kernel_size=1)
        )
        
        def down_block(in_dim,out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim,out_dim,kernel_size=4,stride=2,padding=1,groups=out_dim//16),
                nn.GroupNorm(out_dim//16,out_dim),
                Prod([
                   nn.Conv2d(out_dim,out_dim,3,padding=1,groups=out_dim),
                   nn.Tanh(), 
                ]),
                nn.Dropout2d(dropout),
                nn.SiLU()
            )
        
        def up_block(in_dim,out_dim):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_dim,out_dim,1),
                nn.GroupNorm(out_dim//16,out_dim),
                Prod([
                   nn.Conv2d(out_dim,out_dim,3,padding=1,groups=out_dim),
                   nn.Tanh(), 
                ]),
                nn.SiLU()
            )
        self.down1 = down_block(layer_dims[0],layer_dims[1])
        self.down2 = down_block(layer_dims[1],layer_dims[2])
        self.down3 = down_block(layer_dims[2],layer_dims[3])
        
        self.attn = nn.Sequential(
            SelfAttention(layer_dims[-1],add_rotary_embedding=True,add_absolute_pos=True,dimensions=2,xsa=True),
            nn.GroupNorm(layer_dims[-1]//16,layer_dims[-1]),
            Prod([
                nn.Conv2d(layer_dims[-1],layer_dims[-1],1),
                nn.Tanh(),
            ]),
            nn.Dropout(dropout),
            nn.SiLU(),
        )
        
        self.up1 = up_block(layer_dims[3],layer_dims[2])
        self.up2 = up_block(layer_dims[2],layer_dims[1])
        self.up3 = up_block(layer_dims[1],layer_dims[0])
        
        self.final = nn.Sequential(
            nn.Conv2d(layer_dims[0],compression**2*out_channels,kernel_size=3,padding=1),
            nn.PixelShuffle(compression),
        )
        
    def forward(self,x):
        x = self.compress(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        bottom = self.attn(d3)
        
        u1 = self.up1(bottom)
        u2 = sum_tensors(d1,self.up2(u1))
        u3 = sum_tensors(x,self.up3(u2))
        
        return self.final(u3)
  