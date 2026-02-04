import torch
from torch import nn
from kemsekov_torch.residual import Residual
import torch.nn.functional as F
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from kemsekov_torch.rotary_emb import RotEmb

class AbsoluteRelativePositionalEmbedding(nn.Module):
    """
    Learnable positional embedding with CAPE-inspired augmentation.
    Encodes normalized spatial coordinates via MLP and injects them using
    FiLM-style modulation (x·(1+scale) + shift). Augmentation (shift/scale)
    applied during training encourages relative positional awareness.
    Supports 1D/2D/3D inputs.
    """
    def __init__(self, x_dim,dimensions : Literal[1,2,3] = 2,cape_shift=1.0,cape_scale=2.0,jit_prob = 0.75) -> None:
        """
        Args:
            x_dim: Input channel dimension.
            dimensions: Spatial dimensionality (1, 2 or 3).
            cape_shift: Max uniform shift for position augmentation [-shift, +shift].
            cape_scale: Max scale factor for position augmentation [0, scale].
            jit_prob: probability of applying augmentation
        """
        super().__init__()
        self.dimensions=dimensions
        self.absolute_pos = nn.Sequential(
            nn.Linear(dimensions,x_dim),
            nn.SiLU(),
            nn.Linear(x_dim,2*x_dim),
        )
        self.cape_shift=cape_shift
        self.cape_scale=cape_scale
        self.jit_prob=jit_prob
        self.pos_gamma = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("cached_grid", torch.tensor([0]),persistent=False)
        self.register_buffer("max_dim_size", torch.tensor([0]))
        
    def forward(self,x):
        DIMS = x.shape[2:]
        dims_len = len(DIMS)
        
        if self.training:
            max_dim_size = max(max(DIMS),self.max_dim_size)
            self.max_dim_size+=max_dim_size-self.max_dim_size
        
        if DIMS == self.cached_grid.shape[:-1]:
            POS_IND = self.cached_grid
        else:
            if dims_len==1:
                POS_IND = torch.arange(0,DIMS[0],device=x.device)[:,None]-DIMS[0]/2
            
            if dims_len==2:
                X = torch.arange(0,DIMS[0],device=x.device)-DIMS[0]/2
                Y = torch.arange(0,DIMS[1],device=x.device)-DIMS[1]/2
                POS_IND = torch.stack(torch.meshgrid([Y,X],indexing='ij'),-1)
            
            if dims_len==3:
                X = torch.arange(0,DIMS[0],device=x.device)-DIMS[0]/2
                Y = torch.arange(0,DIMS[1],device=x.device)-DIMS[1]/2
                Z = torch.arange(0,DIMS[2],device=x.device)-DIMS[2]/2
                POS_IND = torch.stack(torch.meshgrid([Z,X,Y],indexing="ij"),-1)

            POS_IND=POS_IND*2
            self.cached_grid = POS_IND
        
        
        # pos ind always centered at zero, and be in range [-1,1]
        POS_IND=POS_IND[None,:]/self.max_dim_size
        
        # apply CAPE Augmentation Transformations 
        if self.training and self.jit_prob>0:
            # for more stable gradients, apply transformations batch-wise
            B = x.shape[0]
            
            #randomly shift in range [-cape_shift,cape_shift]
            pos_ind_shift = (torch.rand([B]+[1]*(POS_IND.ndim-1)+[self.dimensions],device=x.device)*2-1)*self.cape_shift
            #randomly scale in range [0,cape_scale]
            pos_ind_scale = torch.rand([B]+[1]*(POS_IND.ndim-1)+[self.dimensions],device=x.device)*self.cape_scale
            
            # with probability (1-jit) choose batches that are not going to be augmented
            if self.jit_prob<1:
                batches_not_to_augment = torch.rand((B,),device=x.device)>self.jit_prob
                pos_ind_scale[batches_not_to_augment]=1
                pos_ind_shift[batches_not_to_augment]=0
            # shift and scale positions to make attention work on relative positions rather than fixed
            POS_IND=POS_IND*pos_ind_scale+pos_ind_shift
        #apply gamma to make at the training start this transformation work as identity
        pos_scale,pos_shift = (self.pos_gamma*self.absolute_pos(POS_IND)).transpose(1,-1).squeeze(-1).chunk(2,1)
        
        # apply proposed positions embedding in following way
        return x*(1+pos_scale)+pos_shift

def zero_module(module):
    """
    Zero out the parameters of a module and return it to implement Re-Zero
    """
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
    return module

class SelfAttention(nn.Module):
    """
    Self-attention block for vision model bottlenecks.
    Input/Output: [B, C, ...]
    """
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        head_dim: int = 64,
        dropout=0.0,
        dimensions : Literal[1,2,3] = 2,
        add_rotary_embedding = False,
        linear=False
    ):
        """
        dim: input dimensions
        heads: heads count. For ViT 8-12 heads is optimal.
        head_dim: dims per head. For ViT 64 is gold standard.
        dropout: attention dropout. Optimal value depends, but for visual tasks 0.1 is good.
        dimensions: expected input dimensions
        add_rotary_embedding: add rotary embedding to input or not. By default all three spacial resolutions are supported, so proper 1,2,3-dimensional rotary embedding is applied
        linear: whether to use custom-linear attention or not. Current linear attention although works, but is not optimized and default non-linear attention works a lot faster.
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.dropout=dropout
        inner_dim = heads * head_dim
        self.linear = linear
        self.dimensions=dimensions
        
        self.abs_emb = AbsoluteRelativePositionalEmbedding(dim,dimensions,jit_prob=0.75)
        
        self.add_rotary_embedding=add_rotary_embedding
        self.rotary_emb = RotEmb()
        
        # small heuristic for groups number estimation
        groups = max(1,dim//32)
        if groups==1 and dim//16>=2: groups=2
        
        # Pre-normalization with GroupNorm
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=1e-6)
        
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        
        self.to_qkv = conv(dim, inner_dim * 3, 1, bias=True)
        
        # Zero-initialized output projection
        self.to_out = zero_module(conv(inner_dim, dim, 1, bias=True))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, ...] with at most 3 dimensions in `...` part
        Returns: Tensor same shaped as `x` with residual connection
        """
        identity = x
        B = x.shape[0]
        
        # 1. Pre-normalization (GroupNorm)
        x = self.norm(self.abs_emb(x))
        
        # 2. QKV projection via 1x1 conv
        qkv = self.to_qkv(x)

        # 3. Reshape to [B, 3, heads, head_dim, L].
        qkv = qkv.view(B, 3, self.heads, self.head_dim, -1)
        
        if self.add_rotary_embedding:
            
            # 3. Reshape to [B, 3,L, heads, head_dim].
            qkv_perm = qkv.permute(0, 1, 4, 2, 3)
            
            # expand x spatial dimensions to [B, 3, (...dims...), heads, head_dim].
            qkv_perm = qkv_perm.view(B,3,*x.shape[2:],self.heads,self.head_dim)
            
            # apply rotary embedding to query and keys
            qkv_perm[:,0]=self.rotary_emb(qkv_perm[:,0])
            qkv_perm[:,1]=self.rotary_emb(qkv_perm[:,1])
        
        # make qkv contiguous to use faster attention path
        qkv = qkv.transpose(-1,-2).contiguous()

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, L]
        
        # (BATCH_SIZE, ... , HEADS_NUM, LENGTH, HEAD_DIM)
        
        if self.linear:
            attn_out = fast_linear_path_BHDL(q,k,v)
        else:
            # 5. Scaled dot-product attention (uses FlashAttention-2 when available)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout,
                is_causal=False
            )  # [B, heads, L, head_dim]
        
        # 6. Reshape back
        attn_out = attn_out.transpose(-1, -2).reshape(B, self.heads * self.head_dim, *x.shape[2:])
        
        # 7. Output projection + residual connection
        return identity + self.to_out(attn_out)

class CrossAttention(nn.Module):
    """
    Cross-attention block. 
    Queries come from `x`, Keys/Values come from `memory`.
    """
    def __init__(
        self, 
        dim: int, 
        context_dim: Optional[int] = None, # If memory has different channels
        heads: int = 8, 
        head_dim: int = 64,
        dropout=0.0,
        dimensions: Literal[1,2,3] = 2,
        add_rotary_embedding = False,
        linear=False
        
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.linear = linear
        inner_dim = heads * head_dim
        context_dim = context_dim if context_dim is not None else dim
        self.abs_emb = AbsoluteRelativePositionalEmbedding(dim,dimensions)
        
        self.add_rotary_embedding = add_rotary_embedding
        self.rotary_emb = RotEmb()
        
        groups = max(1, dim // 32)
        if groups == 1 and dim // 16 >= 2: groups = 2
        
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=1e-6)
        self.norm_context = nn.GroupNorm(num_groups=max(1, context_dim // 32), num_channels=context_dim, eps=1e-6)
        
        conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dimensions - 1]
        
        self.to_q = conv(dim, inner_dim, 1, bias=True)
        self.to_kv = conv(context_dim, inner_dim * 2, 1, bias=True)
        
        self.to_out = zero_module(conv(inner_dim, dim, 1, bias=True))
        

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        identity = x
        B = x.shape[0]
        
        # 1. Pre-normalization
        x = self.norm(self.abs_emb(x))
        memory = self.norm_context(memory)
        
        # 2. Project Q and KV
        q = self.to_q(x)          # [B, inner_dim, ...]
        kv = self.to_kv(memory)   # [B, inner_dim * 2, ...]

        # 3. Reshape for attention
        q = q.view(B, 1, self.heads, self.head_dim, -1)
        kv = kv.view(B, 2, self.heads, self.head_dim, -1)
        
        if self.add_rotary_embedding:
            # 3. Reshape to [B, 3,L, heads, head_dim].
            kv_perm = kv.permute(0, 1, 4, 2, 3)
            q_perm = q.permute(0, 1, 4, 2, 3)
            
            # expand x spatial dimensions to [B, 3, (...dims...), heads, head_dim].
            kv_perm = kv_perm.view(B,2,*memory.shape[2:],self.heads,self.head_dim)
            q_perm = q_perm.view(B,1,*x.shape[2:],self.heads,self.head_dim)
            
            # apply rotary embedding to query and keys
            kv_perm[:,0]=self.rotary_emb(kv_perm[:,0])
            q_perm[:,0]=self.rotary_emb(q_perm[:,0])
        
        # make qkv contiguous to use faster attention path
        q = q[:,0].transpose(-1,-2).contiguous()
        kv = kv.transpose(-1,-2).contiguous()

        k, v = kv[:, 0], kv[:, 1]
        
        # qkv of shape
        # (BATCH_SIZE, HEADS_NUM, LENGTH, HEAD_DIM)
        
        # use linear attention when needed
        if self.linear:
            attn_out = fast_linear_path_BHDL(q,k,v)
        else:
            # 5. Scaled dot-product attention (uses FlashAttention-2 when available)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout,
                is_causal=False
            )  # [B, heads, L, head_dim]
        
        # 6. Reshape back
        attn_out = attn_out.transpose(-1, -2).reshape(B, self.heads * self.head_dim, *x.shape[2:])
        
        # 7. Output projection + residual connection
        return identity + self.to_out(attn_out)
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

def g(x):
    return torch.nn.functional.softplus(x,5)**2
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


def fast_linear_path_BHDL(q, k, v):
    """
    q, k, v: [B, H, Length, D]
    Output: [B, H, Length, D]
    """
    q_perm = q.permute(0, 2, 1, 3) # [B, H, D, L] -> [B, L, H, D]
    k_perm = k.permute(0, 2, 1, 3)
    v_perm = v.permute(0, 2, 1, 3)
    return fast_linear_path_einsum(q_perm, k_perm, v_perm).permute(0, 2, 1, 3) 

class EfficientSpatialChannelAttention(nn.Module):
    """
    Efficient Spatial Channel Attention (ESCA) Module

    Applies efficient spatial attention across channels using a grouped [1,2,3]D convolution
    over the spatial dimensions. This module computes attention weights
    that modulate channel responses based on spatial structure, improving
    feature representation with minimal overhead.
    """
    def __init__(self, channels,groups='auto',dimensions=2,kernel_size=3):
        super().__init__()
        assert dimensions in [1,2,3],f"dimensions must be one of [1,2,3], but got {dimensions}"
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        
        groups = max(1,channels//32)
        if groups==1: groups=2
        
        self.spatial_attn = nn.Sequential(
            conv(channels,channels,kernel_size,padding=kernel_size//2,groups=groups),
            nn.Tanh()
        )

    def forward(self, x):
        spatian_attn = self.spatial_attn(x)
        return x*spatian_attn