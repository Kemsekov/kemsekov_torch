from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from kemsekov_torch.residual import Residual
import math

from kemsekov_torch.common_modules import ChanLayerNorm

def reshape_to_transformer_input(x : torch.Tensor):
    """
    x of shape [batch,channels,...dims...]
    """
    return x.flatten(2).permute(0,2,1)
def restore_shape_of_transformer_output(out,src_shape : List[int]):
    return out.permute(0,2,1).view(src_shape)

class FlattenSpatialDimensions(nn.Module):
    """
    Prepares vison-like 1d,2d,3d sequential data into format suitable for transformer
    
    Permutes spatial dimension-like input 
    `[batch,channels,dim1,dim2,...]` to `[batch,dim*dim2*...,channels]`
    
    Then feeds this tensor to input module m and reshapes it's output back to original shape.
    """
    def __init__(self, m):
        """
        Permutes spatial dimension-like input 
        `[batch,channels,dim1,dim2,...]` to `[batch,dim*dim2*...,channels]`
        
        Then feeds this tensor to input module m and reshapes it's output back to original shape.
        """
        super().__init__()
        if isinstance(m,list) or isinstance(m,tuple):
            self.m = nn.Sequential(*m)
        else:
            self.m  = m
        
    def forward(self,x):
        x_shape = list(x.shape)
        x_flat = reshape_to_transformer_input(x)
        out = self.m(x_flat)
        x_shape[1] = out.shape[-1] # update channels
        return restore_shape_of_transformer_output(out,torch.Size(x_shape))

# these two modules are kinda legacy, they don't implement anything, just for convenience
class TransformerSelfAttentionBlock(nn.Module):
    """
    Full Self-Attention transformer encoder that accepts tensors of shape [batch,channels,...dims...]
    """
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward = 2048, 
        dropout = 0.1, 
        activation = torch.nn.functional.relu, 
        layer_norm_eps = 0.00001, 
        batch_first = True, 
        norm_first = False, 
        bias = True, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.m = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False
    ):
        out = self.m(src,src_mask,src_key_padding_mask,is_causal)
        return out
class TransformerCrossAttentionBlock(nn.Module):
    """
    Full Cross-Attention transformer decoder that accepts tensors of shape [batch,channels,...dims...]
    """
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward = 2048, 
        dropout = 0.1, 
        activation = nn.functional.relu, 
        layer_norm_eps = 0.00001, 
        batch_first = True, 
        norm_first = False, 
        bias = True, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.m = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        
    def forward(
        self,
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        tgt_mask: torch.Tensor | None = None, 
        memory_mask: torch.Tensor | None = None, 
        tgt_key_padding_mask: torch.Tensor | None = None, 
        memory_key_padding_mask: torch.Tensor | None = None, 
        tgt_is_causal: bool = False, 
        memory_is_causal: bool = False
    ):
        out = self.m(tgt,memory,tgt_mask,memory_mask,tgt_key_padding_mask,memory_key_padding_mask,tgt_is_causal,memory_is_causal)
        return out
class LinearSelfAttentionBlock(torch.nn.Module):
    def __init__(self,input_dim,mlp_dim,heads=8,dropout=0.1,device=None):
        """
        Linear self-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        """
        super().__init__()
        self.attn = LinearCrossAttentionBlock(input_dim,mlp_dim,heads,dropout,device)
    def forward(self,x):
        return self.attn(x,x)
class LinearCrossAttentionBlock(torch.nn.Module):
    def __init__(self,input_dim,mlp_dim,heads=8,dropout=0.1,device=None):
        """
        Linear cross-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        """
        super().__init__()
        self.Q = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
                device=device,
            ),
            nn.LayerNorm(input_dim,device=device)
        )
        
        self.K = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
                device=device,
            ),
            nn.LayerNorm(input_dim,device=device)
        )
        
        self.V = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
                device=device,
            ),
            nn.LayerNorm(input_dim,device=device)
        )

        self.attn = MultiHeadLinearAttention(
            input_dim,
            heads,
            dropout=dropout,
            add_zero_token=True,
            device=device
        )
        self.attn_norm = nn.LayerNorm(input_dim,device=device)
        
        self.mlp=Residual([
            nn.Linear(input_dim,mlp_dim,device=device),
            nn.Dropout(dropout,inplace=True),
            nn.GELU(),
            nn.Linear(mlp_dim,input_dim,device=device),
        ])
        
        self.local_attention = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=5,
            padding=2,
            device=device,
            groups=heads
        )
    
    def _local_attnetion(self,x):
        return x+self.local_attention(x.transpose(-2,-1)).transpose(-2,-1)
    
    def forward(self,query_source : torch.Tensor, context : torch.Tensor):
        """
        Computes multihead cross attention for given context and query source.
        
        query_source: tensor that is used to compute query(Q) of attention. 
        We need to embed information from context in this tensor. Output shape will be equal to query_source shape.
        
        context: tensor that is used to compute keys(K) and values(V) of attention. It is additional information that we need to embed into query_source.
        
        query_source shape can be != context shape, only batch and channel dimensions needs to match.
        
        When context==query_source, the results will be same as self-attention.
        """
        
        #--------------------
        # start = time.time()
        
        Q = self.Q(query_source)
        K,V = self.K(context),self.V(context)
        
        attn = self.attn(Q,K,V)[0]
        attn = self._local_attnetion(attn)
        attn=self.attn_norm(attn)
        
        #--------------------
        # print("total attn",time.time()-start)
        # start = time.time()
        
        result = self.mlp(attn)
        result+=query_source
        #--------------------
        # print("mlp + reshape",time.time()-start)
        
        return result
class LinearAttention(nn.Module):
    """
    Accepts Q,K,V of shapes [batch,heads,seq_length,dim]
    """
    def __init__(self,embed_dim):
        """
        Initialize linear attention module with given emb dim
        
        Accepts Q,K,V of shapes [batch,heads,seq_length,dim]
        """
        super().__init__()
        self.embed_dim=embed_dim

    def forward(self,Q,K,V,phi_Q,phi_K,compute_attn_weight  : bool = False):
        """
        phi_Q,phi_K is produced from Q and K via kernel
        """
        phi_K = phi_K.transpose(-2,-1)
        K = K.transpose(-2,-1)
        embed_dim = Q.shape[-1]
        seq_len = Q.shape[-2]
        
        # here we apply RALA-like approach to increase phi_q @ phi_k matrix rank by rescaling each sample
        # compute global query
        
        q_global = torch.mean(Q,-2,keepdim=True)/(embed_dim**0.5)
        alpha = (q_global @ phi_K).softmax(-1)*seq_len
        phi_K=alpha*phi_K
        
        # the full version linear attention
        if compute_attn_weight:
            linear_attn = (phi_Q @ phi_K)
            linear_attn = linear_attn/(linear_attn.sum(-1,keepdim=True) + 1e-6)
        else:
            linear_attn = None
        
        # rearanged attention version that have linear complexity
        K_sum = phi_K.sum(-1,keepdim=True)
        KV = phi_K @ V
        linear_out_fast = (phi_Q @ KV)/(phi_Q @ K_sum + 1e-6)

        return linear_out_fast,linear_attn

class AddConst(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c = c
    def forward(self,x):
        return x+self.c

class MultiHeadLinearAttention(nn.Module):
    """
    Multi‐head wrapper around single‐head LinearAttention, allowing different
    sequence lengths for Q vs. K/V (i.e. cross‐attention).
    
    - embed_dim = n_heads * head_dim
    - Q: [batch, L_Q,  embed_dim]
    - K: [batch, L_K,  embed_dim]
    - V: [batch, L_K,  embed_dim]
    Returns:
      - output: [batch, L_Q,  embed_dim]
      - attn:   [batch, n_heads, L_Q, L_K]   (if compute_attn_weight=True)
    """
    def __init__(self, embed_dim, n_heads,dropout = 0.0,add_zero_token = False,device = None):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.kernel_Q = nn.Sequential(
            nn.Linear(embed_dim,embed_dim,device=device),
            nn.Tanh(),
            AddConst(1)
        )
        
        self.kernel_K = nn.Sequential(
            nn.Linear(embed_dim,embed_dim,device=device),
            nn.Tanh(),
            AddConst(1)
        )
        
        self.feature_dropout = nn.Dropout(dropout, inplace=True)
        
        self.add_zero_token=add_zero_token
        if add_zero_token:
            self.zero_token = nn.Parameter(torch.zeros(1, 1, embed_dim,device=device), requires_grad=False)
        
        self.single_head_attn = LinearAttention(self.head_dim)
        
    def split_heads(self, x : torch.Tensor):
        # x: [B, seq_len, embed_dim]
        B = x.shape[0]
        x = x.view(B, -1, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).view(B, self.n_heads, -1, self.head_dim)
    
    def forward(self, Q, K, V, compute_attn_weight : bool = False):
        """
        Q: [B, L_Q,  embed_dim]
        K: [B, L_K,  embed_dim]
        V: [B, L_K,  embed_dim]
        """
        if self.add_zero_token:
            Z = self.zero_token.expand(Q.shape[0], 1, -1)  # (batch,1,dim)
            K = torch.cat([Z, K], dim=1)
            V = torch.cat([Z, V], dim=1)
        
        Q = self.feature_dropout(Q)
        K = self.feature_dropout(K)
        
        B, L_Q, _ = Q.shape
        _, L_K, _ = K.shape
        
        phi_Q = self.kernel_Q(Q)
        phi_K = self.kernel_K(K)
        
        phi_Qh_flat = self.split_heads(phi_Q)   # → [B * n_heads, L_Q, head_dim]
        phi_Kh_flat = self.split_heads(phi_K)   # → [B * n_heads, L_K, head_dim]
        Qh_flat = self.split_heads(Q)   # → [B * n_heads, L_Q, head_dim]
        Kh_flat = self.split_heads(K)   # → [B * n_heads, L_K, head_dim]
        Vh_flat = self.split_heads(V)   # → [B * n_heads, L_K, head_dim]
        
        # 3. Run single‐head linear attention
        out_flat, attn_flat = self.single_head_attn(
            Qh_flat, Kh_flat, Vh_flat,phi_Qh_flat,phi_Kh_flat, compute_attn_weight
        )
        # out_flat: [B * n_heads, L_Q, head_dim]
        # attn_flat (if requested): [B * n_heads, L_Q, L_K]

        # 4. Un‐flatten heads
        out_heads = out_flat.view(B, self.n_heads, L_Q, self.head_dim)
        # → [B, n_heads, L_Q, head_dim]
        out_heads = out_heads.permute(0, 2, 1, 3).contiguous()
        # → [B, L_Q, n_heads, head_dim]
        output = out_heads.view(B, L_Q, self.embed_dim)
        # → [B, L_Q, embed_dim]

        if attn_flat is not None:
            attn = attn_flat.view(B, self.n_heads, L_Q, L_K)
        else:
            attn = None

        return output, attn

class EfficientSpatialChannelAttention(nn.Module):
    """
    Efficient Spatial Channel Attention (ESCA) Module

    Applies efficient spatial attention across channels using a 1D convolution
    over the flattened spatial dimensions. This module computes attention weights
    that modulate channel responses based on spatial structure, improving
    feature representation with minimal overhead.

    Parameters
    ----------
    channels : int
        Number of input channels (C).
    ks : int, optional
        Kernel size for the 1D convolution used in spatial attention,
        by default 5. Must be an odd number for symmetric padding.

    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape [N, C, *spatial_dims], where spatial_dims can be 1D, 2D, or 3D.

    Output Shape
    ------------
    out : torch.Tensor
        Tensor of same shape as input, with spatially modulated channel responses.

    Example
    -------
    >>> module = EfficientSpatialChannelAttention(channels=64, ks=5)
    >>> x1d = torch.randn(8, 64, 128)
    >>> x2d = torch.randn(8, 64, 32, 32)
    >>> x3d = torch.randn(8, 64, 8, 16, 16)
    >>> y1d = module(x1d)
    >>> y2d = module(x2d)
    >>> y3d = module(x3d)
    """
    def __init__(self, channels,ks=5):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(channels,channels,ks,padding=ks//2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [N, C, ...]
        N, C = x.shape[:2]

        # Global Average Pooling over spatial dims to [N, C, 1]
        flat = x.view(N,C,-1)
        spatian_attn = self.spatial_attn(flat)
        
        # out = torch.max(flat*ch_attn,flat*spatian_attn)
        out = flat*spatian_attn
        return out.view(x.shape)