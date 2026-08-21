from kemsekov_torch.common_modules import resize_tensor, Prod
from kemsekov_torch.attention import SelfAttention
import torch.nn as nn

def sum_tensors(a,b):
    if a.shape!=b.shape:
        b = resize_tensor(b,a.shape[1:])
    return a+b

class Unet(nn.Module):
    def __init__(self, in_channels,out_channels,compression = 4,layer_dims = [32,64,128,256],dropout=0):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        
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
        downs = []
        for prev_l,next_l in zip(layer_dims,layer_dims[1:]):
            downs.append(down_block(prev_l,next_l))
        self.downs = nn.ModuleList(downs)
        
        self.attn = nn.Sequential(
            SelfAttention(layer_dims[-1],add_rotary_embedding=True,add_absolute_pos=False,dimensions=2,xsa=True),
            nn.GroupNorm(layer_dims[-1]//16,layer_dims[-1]),
            Prod([
                nn.Conv2d(layer_dims[-1],layer_dims[-1],1),
                nn.Tanh(),
            ]),
            nn.Dropout(dropout),
            nn.SiLU(),
        )
        ups = []
        for prev_l,next_l in reversed(list(zip(layer_dims,layer_dims[1:]))):
            ups.append(up_block(next_l,prev_l))
        self.ups = nn.ModuleList(ups)
        
        self.final = nn.Sequential(
            nn.Conv2d(layer_dims[0],compression**2*out_channels,kernel_size=3,padding=1),
            nn.PixelShuffle(compression),
        )
        
    def forward(self,x):
        x = self.compress(x)
        
        down_act = []
        d = x
        for down in self.downs:
            down_act.append(d)
            d = down(d)
        
        u = self.attn(d)
        # ups = [u]
        for up,d in zip(self.ups,reversed(down_act)):
            u = sum_tensors(d,up(u))
            # ups.append(u)
        
        # print("DOWNS")
        # for d in down_act:
        #     print(d.shape)
        # print("UPS")
        # for d in ups:
        #     print(d.shape)
        
        return self.final(u)
        
        x = self.compress(x)
        
        d1 = self.downs[0](x)
        d2 = self.downs[1](d1)
        d3 = self.downs[2](d2)
        
        bottom = self.attn(d3)
        
        u1 = sum_tensors(d2,self.ups[0](bottom))
        u2 = sum_tensors(d1,self.ups[1](u1))
        u3 = sum_tensors(x,self.ups[2](u2))
        
        return self.final(u3)
  