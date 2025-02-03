import torch
import torch.nn as nn
from positional_emb import PositionalEncodingPermute
from residual import ResidualBlock

class VisualMultiheadSelfAttentionPixelwise(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=8,dropout_p=0.1):
        super().__init__()
        
        self.out_channels = out_channels
        # accepts BATCH_SIZE,SEQ_LENGTH,EMB_DIM
        self.attn = torch.nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=out_channels,
            vdim=out_channels,
            kdim=out_channels,
            dropout=dropout_p,
            batch_first=True
        )

        self.Q = ResidualBlock(in_channels,out_channels)
        self.K = ResidualBlock(in_channels,out_channels)
        self.V = ResidualBlock(in_channels,out_channels)
        
        self.inp_pos_enc = PositionalEncodingPermute(in_channels)
        
        if in_channels!=out_channels:
            self.x_residual = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.x_residual = nn.Identity()
        
    def forward(self,x : torch.Tensor):
        x_res = self.x_residual(x)
        x = self.inp_pos_enc(x)+x
        # add positional encoding to input image
        Q = self.Q(x).flatten(2).transpose(-1,-2)
        K = self.K(x).flatten(2).transpose(-1,-2)
        V = self.V(x)
        
        out_shape = list(V.shape)
        
        # batch,out_channels,width,height
        out_shape[1]=self.out_channels
        
        V=V.flatten(2).transpose(-1,-2)
        # compute self-attention of input image
        out_flat,b = self.attn(V,Q,K)
        out = out_flat.transpose(-1,-2).view(out_shape)
        
        return out+x_res
        