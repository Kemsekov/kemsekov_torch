import torch
from torch import nn
from kemsekov_torch.residual import Residual
from kemsekov_torch.common_modules import AddConst
from kemsekov_torch.rotary_emb import RotaryEmbHeadsInplace
import torch.nn.functional as F

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
    def __init__(self,input_dim,mlp_dim,heads=8,dropout=0.1,device=None,activation=torch.nn.GELU,add_rotary_emb=False,add_zero_token=False):
        """
        Accepts inputs of size [batch, ... ,  embed_dim] where (...) is spatial dimensions up to 3
        
        Linear self-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        
        activation: what activation function to use
        
        add_zero_token: add learned zero token to input or not
        """
        super().__init__()
        self.attn = LinearCrossAttentionBlock(input_dim,mlp_dim,heads,dropout,device,activation,add_rotary_emb=add_rotary_emb,add_zero_token=add_zero_token)
    def forward(self,x):
        return self.attn(x,x)
from kemsekov_torch.rotary_emb import RotaryEmbHeadsInplace
class LinearCrossAttentionBlock(torch.nn.Module):
    def __init__(self,input_dim,mlp_dim,heads=8,dropout=0.1,device=None,activation=torch.nn.GELU,add_rotary_emb=False,add_zero_token=False):
        """
        Accepts inputs of size [batch, ... ,  embed_dim] where (...) is spatial dimensions up to 3
        
        Linear cross-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        
        activation: what activation function to use
        
        add_zero_token: add learned zero token to input or not
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
            add_zero_token=add_zero_token,
            device=device,
            add_rotary_emb=add_rotary_emb
        )
        self.attn_norm = nn.LayerNorm(input_dim,device=device)
        
        self.mlp=Residual([
            nn.Linear(input_dim,mlp_dim,device=device),
            nn.Dropout(dropout,inplace=True),
            activation(),
            nn.Linear(mlp_dim,input_dim,device=device),
        ])
        
        # self.local_attention_gamma = torch.nn.Parameter(torch.tensor(0.0))
        # self.local_attention = nn.Conv1d(
        #     input_dim,
        #     input_dim,
        #     kernel_size=5,
        #     padding=2,
        #     device=device,
        #     groups=heads
        # )
    
    def _local_attnetion(self,x):
        # x: [batch, ... ,channels]
        xt = x
        if len(x.shape)>3:
            batch = xt.shape[0]
            ch = xt.shape[-1]
            # xt: [batch,(...),channels]
            xt = xt.view(batch,-1,ch)
        
        # xt: [batch,channels,(...)]
        xt = xt.transpose(-2,-1)
        
        # out: [batch,channels,(...)]
        out = self.local_attention_gamma*self.local_attention(xt)
        
        # out: [batch,(...),channels]
        out = out.transpose(-2,-1)
        
        if len(x.shape)>3:
            # out: [batch, ... ,channels]
            out = out.view(x.shape)
        
        return out
    
    
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
        # attn = attn+self._local_attnetion(attn)
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
class EluKernel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.nn.functional.elu(x)+1
class LogKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        c = self.c.abs()+1e-6
        return torch.log(1+torch.exp(-x*c))/c+x
class XReLUKernel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return torch.relu(x)*x

def compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        q, k, v: Tensors of shape [batch_size, seq_length, num_heads, head_dim]

    Returns:
        Tensor of shape [batch_size, seq_length, num_heads, head_dim]
    """

    # Permute to [batch_size, num_heads, seq_length, head_dim]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # Compute attention
    attn_output = F.scaled_dot_product_attention(q, k, v)

    # Permute back to [batch_size, seq_length, num_heads, head_dim]
    attn_output = attn_output.permute(0, 2, 1, 3)

    return attn_output

def g(x):
    return torch.relu(x)*x

# Fast linear attention with inputs of shape [B, S, H, D] using einsum
def fast_linear_path_einsum(q, k, v):
    """
    q, k, v: Tensors of shape [B, S, H, D]
    Returns:
        out_fast: Tensor of shape [B, S, H, D]
    """
    B, L, H, D = q.shape
    # RALA rescaling: compute global q mean over sequence dim
    q_global = q.mean(dim=1, keepdim=True) / (D ** 0.5)  # [B, 1, H, D]

    # Compute scaling alpha: dot(q_global, K) over D -> [B, 1, H, L]
    alpha = torch.einsum('bihd,blhd->bihl', q_global, k).softmax(dim=-1) * L  # [B, 1, H, L]

    # Broadcast alpha to match phi_K: reshape to [B, L, H, 1]
    alpha_reshaped = alpha.squeeze_(1).permute(0, 2, 1).unsqueeze_(-1)  # [B, L, H, 1]
    k = k * alpha_reshaped  # [B, L, H, D]
    
    # Compute gated features
    g_q  = g(q)      # [B, S, H, D]
    g_k  = g(k)      # [B, S, H, D]
    g_mq = g(-q)     # [B, S, H, D]
    g_mk = g(-k)     # [B, S, H, D]

    # term1: g_q @ (g_k^T @ v)
    # first compute g_k^T @ v via sum over sequence dim
    # g_k: [B, S, H, D], v: [B, S, H, D]
    # -> gk_v: [B, H, D, D]
    gk_v = torch.einsum('bshd, bshv -> bhdv', g_k, v)
    # then contract with g_q over feature dim
    # g_q: [B, S, H, D], gk_v: [B, H, D, D]
    term1 = torch.einsum('bshd, bhdv -> bshv', g_q, gk_v)

    # term2: same for negatives
    gmk_v = torch.einsum('bshd, bshv -> bhdv', g_mk, v)
    term2 = torch.einsum('bshd, bhdv -> bshv', g_mq, gmk_v)

    # denominator1: g_q @ sum_s g_k
    # sum over sequence: sum_gk: [B, H, D]
    sum_gk = g_k.sum(dim=1)
    den1 = torch.einsum('bshd, bhd -> bsh', g_q, sum_gk).unsqueeze_(-1)

    # denominator2: same for negatives
    sum_gmk = g_mk.sum(dim=1)
    den2 = torch.einsum('bshd, bhd -> bsh', g_mq, sum_gmk).unsqueeze_(-1)

    # combine
    out_fast = (term1 + term2) / (den1 + den2+1e-6)
    return out_fast


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
    def __init__(self, embed_dim, n_heads,dropout = 0.0,add_zero_token = False,device = None,add_rotary_emb = False):
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
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.feature_dropout = nn.Dropout(dropout, inplace=True)
        self.add_zero_token=add_zero_token
        
        self.zero_token_K = nn.Parameter(torch.zeros(1, 1, embed_dim,device=device), requires_grad=False)
        self.zero_token_V = nn.Parameter(torch.zeros(1, 1, embed_dim,device=device), requires_grad=False)
        
        self.single_head_attn = LinearAttention(self.head_dim)
        
        self.add_rotary_emb=add_rotary_emb
        self.rotary_emb = RotaryEmbHeadsInplace(self.head_dim,freqs_for='pixel')
        
        self.g=nn.Sequential(
            nn.Linear(self.head_dim,self.head_dim),
            nn.LayerNorm(self.head_dim),
            # TanhKernel()
            LogKernel()
            # EluKernel()
        )
        
    def split_heads(self, x : torch.Tensor):
        # x: [B, seq_len, embed_dim]
        B = x.shape[0]
        D = x.shape[-1]
        return x.view(B, -1, self.n_heads, D//self.n_heads)
    
    def permute_to_rotary_input(self,x):
        # [B, ..., embed_dim] -> [B, ...,n_heads, embed_dim]
        x = x.view(list(x.shape[:-1]) + [self.n_heads, -1])
        # [B, ..., n_heads, embed_dim] -> [B, n_heads, ..., embed_dim]
        dims = list(range(len(x.shape)))[1:-2]
        x = x.permute([0,-2]+dims+[-1])
        return x
    
    def unpermute_from_rotary_output(self,x):
        # [B, n_heads, ..., embed_dim] -> [B, ..., n_heads, embed_dim]
        dim = x.shape[-1]
        dims = list(range(len(x.shape)))[2:-1]
        x = x.permute([0]+dims+[1,-1])
        return x.view(x.shape[0],-1,self.n_heads,dim)
    
    def split_heads_with_rotary_emb(self, x : torch.Tensor):
        x = self.permute_to_rotary_input(x)
        x = self.rotary_emb([x])[0]
        x = self.unpermute_from_rotary_output(x)
        return x
    
    def forward(self, Q, K, V, compute_attn_weight : bool = False):
        """
        Q: [B, L_Q,  embed_dim]
        K: [B, L_K,  embed_dim]
        V: [B, L_K,  embed_dim]
        """
        v_dim = V.shape[-1]
        attn = None
        
        Q = self.feature_dropout(Q)
        K = self.feature_dropout(K)
        
        # K, V = self.add_zero_token_KV(K, V)
        
        if self.add_rotary_emb:
            Qh = self.split_heads_with_rotary_emb(Q)
            Kh = self.split_heads_with_rotary_emb(K)
        else:
            Qh = self.split_heads(Q)   # → [B, L_Q, n_heads, head_dim]
            Kh = self.split_heads(K)   # → [B, L_K, n_heads, head_dim]
        Vh = self.split_heads(V)   # → [B, L_K, n_heads, head_dim]
 
        # phi_Qh = self.split_heads(self.kernel_Q(Q))   # → [B, L_K, n_heads, head_dim]
        # phi_Kh = self.split_heads(self.kernel_K(K))   # → [B, L_K, n_heads, head_dim]
        
        Kh, Vh = self.add_zero_token_KhVh(Kh, Vh)
        phi_Qh = self.g(Qh) 
        phi_Kh = self.g(Kh) 
        
        
        # 3. Run single‐head linear attention
        out_heads, attn = self.single_head_attn(
            Qh, Kh, Vh, phi_Qh,phi_Kh, compute_attn_weight
        )
        # out_heads = fast_linear_path_einsum(Qh,Kh,Vh)
        
        # we can try to use full scaled dot product attention to compare results
        # out_heads = compute_attention(Qh,Kh,Vh)
        
        output = out_heads.reshape(list(Q.shape[:-1]) + [v_dim])

        if attn is not None:
            attn = attn.permute([0,2,1,3])
        else:
            attn = None

        return output, attn

    def add_zero_token_KhVh(self, Kh, Vh):
        if self.add_zero_token:
            shape = [Kh.shape[0]]+[1,-1]
            ZK = self.zero_token_K.expand(shape)  # (batch,1,dim)
            ZV = self.zero_token_V.expand(shape)  # (batch,1,dim)
            
            ZK = self.split_heads(ZK)
            ZV = self.split_heads(ZV)
            Kh = torch.cat([ZK, Kh], dim=1)
            Vh = torch.cat([ZV, Vh], dim=1)
        return Kh,Vh

    def add_zero_token_KV(self, K, V):
        if self.add_zero_token:
            shape = [K.shape[0]]+[1]+list(K.shape[2:])
            ZK = self.zero_token_K.expand(shape)  # (batch,1,dim)
            ZV = self.zero_token_V.expand(shape)  # (batch,1,dim)
            
            K = torch.cat([ZK, K], dim=1)
            V = torch.cat([ZV, V], dim=1)
        return K,V
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
            nn.Tanh()
        )

    def forward(self, x):
        # x: [N, C, ...]
        N, C = x.shape[:2]

        # Global Average Pooling over spatial dims to [N, C, 1]
        flat = x.view(N,C,-1)
        spatian_attn = 1+self.spatial_attn(flat)
        
        # out = torch.max(flat*ch_attn,flat*spatian_attn)
        out = flat*spatian_attn
        return out.view(x.shape)