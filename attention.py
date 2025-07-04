import math
import torch
from torch import nn
from kemsekov_torch.residual import Residual
from kemsekov_torch.common_modules import AddConst

# these two modules are kinda legacy, they don't implement anything, just for convenience
class TransformerSelfAttentionBlock(nn.Module):
    """
    Full Self-Attention transformer encoder that accepts tensors of shape [batch,length,channels]
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
    Full Cross-Attention transformer decoder that accepts tensors of shape [batch,length,channels]
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
    def __init__(self,input_dim,mlp_dim,heads=8,dropout=0.1,device=None,activation=torch.nn.GELU):
        """
        Accepts inputs of size [batch, L_Q,  embed_dim]
        
        Linear self-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        
        activation: what activation function to use
        """
        super().__init__()
        self.attn = LinearCrossAttentionBlock(input_dim,mlp_dim,heads,dropout,device,activation)
    def forward(self,x):
        return self.attn(x,x)
from kemsekov_torch.rotary_emb import RotaryEmbHeadsInplace
class LinearCrossAttentionBlock(torch.nn.Module):
    def __init__(self,input_dim,mlp_dim,heads=8,dropout=0.1,device=None,activation=torch.nn.GELU):
        """
        Accepts inputs of size [batch, L_Q,  embed_dim]
        
        Linear cross-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        
        activation: what activation function to use
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
            activation(),
            nn.Linear(mlp_dim,input_dim,device=device),
        ])
        
        self.local_attention_gamma = torch.nn.Parameter(torch.tensor(0.0))
        self.local_attention = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=5,
            padding=2,
            device=device,
            groups=heads
        )
        self.rot_emb = RotaryEmbHeadsInplace(input_dim)
    
    def _local_attnetion(self,x):
        # x: [batch, ... ,channels]
        xt = x
        
        # batch = xt.shape[0]
        # ch = xt.shape[-1]
        
        # xt: [batch,(...),channels]
        # xt = xt.view(batch,-1,ch)
        
        # xt: [batch,channels,(...)]
        xt = xt.transpose(-2,-1)
        
        # out: [batch,channels,(...)]
        out = self.local_attention_gamma*self.local_attention(xt)
        
        # out: [batch,(...),channels]
        out = out.transpose(-2,-1)
        
        # out: [batch, ... ,channels]
        # out = out.view(x.shape)
        
        return out
    
    def add_rotary_emb(self,Q,K):
        Q,K = self.rot_emb.forward_multiple([Q[:,None],K[:,None]])
        Q=Q[:,0]
        K=K[:,0]
        return Q,K
    
    def flatten_qkv(self,Q,K,V):
        batch = Q.shape[0]
        qk_dim = Q.shape[-1]
        v_dim = V.shape[-1]
        Qf = Q.view(batch,-1,qk_dim)
        Kf = K.view(batch,-1,qk_dim)
        Vf = V.view(batch,-1,v_dim)
        return Qf,Kf,Vf
    
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
        Q, K, V = self.Q(query_source),self.K(context),self.V(context)
        # Q,K = self.add_rotary_emb(Q,K)
        # Q,K,V = self.flatten_qkv(Q,K,V)
        
        attn = self.attn(Q,K,V)[0]
        attn = attn+self._local_attnetion(attn)
        attn=self.attn_norm(attn)
        
        #--------------------
        # print("total attn",time.time()-start)
        # start = time.time()
        
        result = self.mlp(attn)#.view(query_source.shape)
        result+=query_source
        #--------------------
        # print("mlp + reshape",time.time()-start)
        
        return result


class LinearAttention(nn.Module):
    """
    Linear attention with RALA-style rescaling,
    Accepts inputs of shape [B, seq_len, H, D] and internally reshapes to [B, H, seq_len, D].
    Works with inputs where heads are the third dimension.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-6

    def forward(self, Q, K, V, phi_Q, phi_K, compute_attn_weight: bool = False):
        # Expect Q, K, V, phi_Q, phi_K shapes: [B, L, H, D]
        B, L, H, D = Q.shape

        # RALA rescaling: compute global q mean over sequence dim
        q_global = Q.mean(dim=1, keepdim=True) / (self.embed_dim ** 0.5)  # [B, 1, H, D]

        # Compute scaling alpha: dot(q_global, K) over D -> [B, 1, H, L]
        alpha = torch.einsum('bihd,blhd->bihl', q_global, K).softmax(dim=-1) * L  # [B, 1, H, L]

        # Broadcast alpha to match phi_K: reshape to [B, L, H, 1]
        alpha_reshaped = alpha.squeeze_(1).permute(0, 2, 1).unsqueeze_(-1)  # [B, L, H, 1]
        phi_K_scaled = phi_K * alpha_reshaped  # [B, L, H, D]

        # Optional full attention weights: [B, L, H, L]
        if compute_attn_weight:
            linear_attn = torch.einsum('blhd,bmhd->blhm', phi_Q, phi_K_scaled)
            linear_attn = linear_attn / (linear_attn.sum(dim=-1, keepdim=True) + self.eps)
        else:
            linear_attn = None

        # Linear path
        # Sum over keys: [B, H, D]
        K_sum = phi_K_scaled.sum(dim=1)  # [B, H, D]
        # KV: sum over sequence for outer-product: [B, H, D, D]
        KV = torch.einsum('blhd,blhe->bhde', phi_K_scaled, V)
        # numerator: phi_Q dot KV over D -> [B, L, H, D]
        numerator = torch.einsum('blhd,bhde->blhe', phi_Q, KV)
        # denominator: phi_Q dot K_sum over D -> [B, L, H, 1]
        denominator = torch.einsum('blhd,bhd->blh', phi_Q, K_sum).unsqueeze_(-1) + self.eps
        out = numerator / denominator  # [B, L, H, D]
        return out, linear_attn

class TanhKernel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.nn.functional.tanh(x)+1

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
        self.feature_dropout = nn.Dropout(dropout, inplace=True)
        self.add_zero_token=add_zero_token
        if add_zero_token:
            self.zero_token = nn.Parameter(torch.zeros(1, 1, embed_dim,device=device), requires_grad=False)
        self.single_head_attn = LinearAttention(self.head_dim)
        
        self.kernel_Q = nn.Sequential(
            nn.Linear(embed_dim,embed_dim,device=device),
            TanhKernel()
        )
        
        self.kernel_K = nn.Sequential(
            nn.Linear(embed_dim,embed_dim,device=device),
            TanhKernel()
        )
        
        # self.kernel_Q = nn.Identity()
        # self.kernel_K = nn.Identity()
    
    def split_heads(self, x : torch.Tensor):
        # x: [B, seq_len, embed_dim]
        B = x.shape[0]
        return x.view(B, -1, self.n_heads, self.head_dim)
    
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
        phi_Q = self.kernel_Q(Q)
        phi_K = self.kernel_K(K)
        
        B, L_Q, _ = Q.shape
        _, L_K, _ = K.shape
        
        Qh = self.split_heads(Q)   # → [B, n_heads, L_Q, head_dim]
        Kh = self.split_heads(K)   # → [B, n_heads, L_K, head_dim]
        Vh = self.split_heads(V)   # → [B, n_heads, L_K, head_dim]
        
        phi_Qh = self.split_heads(phi_Q)   # → [B, n_heads, L_K, head_dim]
        phi_Kh = self.split_heads(phi_K)   # → [B, n_heads, L_K, head_dim]
        
        # 3. Run single‐head linear attention
        out_heads, attn = self.single_head_attn(
            Qh, Kh, Vh,phi_Qh,phi_Kh, compute_attn_weight
        )
        
        # [B, L_Q,n_heads, head_dim]
        # out_heads
        
        output = out_heads.reshape(B, L_Q, -1)

        if attn is not None:
            attn = attn.permute([0,2,1,3])
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