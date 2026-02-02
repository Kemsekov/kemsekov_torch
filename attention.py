import torch
from torch import nn
from kemsekov_torch.residual import Residual
import torch.nn.functional as F

class LinearSelfAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        heads=8,
        dropout=0.1,
        device=None,
        local_attention_dimensions=-1,
        add_zero_token=False,
        add_rotary_emb=False,
        rotary_emb_base=10000,
        use_classic_attention=False,
        add_gating = True
    ):
        """
        Accepts inputs of size [batch, ... ,  embed_dim] where (...) is spatial dimensions up to 3
        
        Linear self-attention block
        
        input_dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dropout: dropout to apply to attention layer
        
        device: where to locate module
        
        activation: what activation function to use
        
        local_attention_dimensions: number of dimensions added when using local attention. If -1 passed, no local attention is added.
        
        add_zero_token: add learned zero token to input or not
        
        add_rotary_emb: add rotary embedding to inputs or not
        
        rotary_emb_base: rotary emb base
        """
        super().__init__()
        self.attn = LinearCrossAttentionBlock(input_dim,heads,dropout,device,local_attention_dimensions=local_attention_dimensions,add_rotary_emb=add_rotary_emb,add_zero_token=add_zero_token,rotary_emb_base=rotary_emb_base,use_classic_attention=use_classic_attention,add_gating=add_gating)
    def forward(self,x):
        return self.attn(x,x)

class _LocalAttention(torch.nn.Module):
    def __init__(self,input_dim,groups,dimensions=1,device=None) -> None:
        super().__init__()
        assert dimensions in [1,2,3]
        self._local_attention_dimensions=dimensions
        self.local_attention = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1](
            input_dim,
            input_dim,
            kernel_size=3,
            padding=1,
            device=device,
            groups=groups
        )
    def forward(self,x):
        # x: [batch, ... ,channels]
        xt = x
        # xt: [batch,channels,(...)]
        xt = xt.transpose(1,-1)
        
        while xt.ndim-2<self._local_attention_dimensions:
            xt=xt.unsqueeze(-1)
        
        # out: [batch,channels,(...)]
        out = torch.tanh(self.local_attention(xt))
        
        # out: [batch,(...),channels]
        out = out.transpose(1,-1)
        
        # out: [batch, ... ,channels]
        out = out.view(x.shape)
        
        return out

class LinearCrossAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        heads=8,
        dropout=0.1,
        device=None,
        local_attention_dimensions=-1,
        add_zero_token=False,
        add_rotary_emb=False,
        rotary_emb_base=10000,
        use_classic_attention=False,
        add_gating = True
    ):
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
        
        add_local_attention: add local attention over output of linear attention or not
        
        local_attention_dimensions: number of dimensions added when using local attention. If -1 passed, no local attention is added. Adds local attention to raw attention outputs via 3x3 convolutions. If your input data is multidimensional, pass proper number of dimensions
        
        add_zero_token: add learned zero token to input or not
        
        add_rotary_emb: add rotary embedding to inputs or not
        
        rotary_emb_base: rotary emb base
        """
        super().__init__()
        
        self.Q = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
                device=device,
            ),
        )
        
        self.K = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
                device=device,
            ),
        )
        
        self.V = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
                device=device,
            ),
        )

        self.attn = MultiHeadLinearAttention(
            input_dim,
            heads,
            dropout=dropout,
            add_zero_token=add_zero_token,
            device=device,
            add_rotary_emb=add_rotary_emb,
            rotary_emb_base=rotary_emb_base,
            use_classic_attention=use_classic_attention
        )
        self.attn_combine = nn.Sequential(
            nn.Linear(2*input_dim,input_dim),
        )
        self.attn_norm = nn.RMSNorm(input_dim,device=device)
        self.mlp=Residual([
            nn.RMSNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim,input_dim,device=device),
        ])

        self.scale = nn.Sequential(
            nn.Linear(input_dim,input_dim),
            nn.RMSNorm(input_dim),
            nn.Tanh()
        )
        
        self._add_local_attention=local_attention_dimensions>0
        self.add_gating=add_gating
        
        if self._add_local_attention:
            self.local_attention = _LocalAttention(input_dim,heads,local_attention_dimensions)
        else:
            self.local_attention = torch.nn.Identity()
    
    
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
        K = self.K(context)
        V = self.V(context)
        
        attn = self.attn(Q,K,V)[0]
        attn = self.attn_combine(torch.concat([self.attn_norm(attn),query_source],-1))
        
        
        attn=self.mlp(attn)
        
        if self._add_local_attention:
            attn = attn*self.local_attention(attn)
        
        if self.add_gating:
            attn=attn*self.scale(attn)

        return attn+query_source


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

from kemsekov_torch.rotary_emb import RotEmb
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
    def __init__(self, embed_dim, n_heads,dropout = 0.0,add_zero_token = False,device = None,add_rotary_emb = False,rotary_emb_base=10000,use_classic_attention=False):
        """
        Multi‐head wrapper around single‐head LinearAttention, allowing different
        sequence lengths for Q vs. K/V (i.e. cross‐attention).
        
        use_classic_attention: for debug purposes, replaces linear attention kernel with scaled dot product attention
        
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
        self.use_classic_attention=use_classic_attention
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.feature_dropout = nn.Dropout(float(dropout), inplace=True)
        self.add_zero_token=add_zero_token
        
        self.zero_token_K = nn.Parameter(torch.zeros(1, 1, embed_dim,device=device), requires_grad=False)
        self.zero_token_V = nn.Parameter(torch.zeros(1, 1, embed_dim,device=device), requires_grad=False)
        
        self.single_head_attn = LinearAttention(self.head_dim)
        
        self.add_rotary_emb=add_rotary_emb
        self.rotary_emb = RotEmb(rotary_emb_base)
        
        # self.g=nn.Sequential(
        #     # nn.Linear(self.head_dim,self.head_dim),
        #     nn.LayerNorm(self.head_dim),
        #     TanhKernel()
        # )
        # self.phi = nn.Sequential(
        #     nn.Linear(embed_dim,embed_dim),
        #     nn.LayerNorm(embed_dim),
        #     # TanhKernel()
        #     LogKernel()
        # )
        
    def split_heads(self, x : torch.Tensor):
        # x: [B, seq_len, embed_dim]
        B = x.shape[0]
        D = x.shape[-1]
        x = x.view(B, -1, self.n_heads, D//self.n_heads)
        
        return x
    
    def split_heads_with_rot_emb(self, x : torch.Tensor):
        # x: [B, seq_len, embed_dim]
        B = x.shape[0]
        D = x.shape[-1]
        dims = list(x.shape[1:-1])
        x = x.view([B] + dims + [self.n_heads, D//self.n_heads])
        x = self.rotary_emb(x)
        x = x.view(B,-1,self.n_heads, D//self.n_heads)
        
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
            Qh = self.split_heads_with_rot_emb(Q)
            Kh = self.split_heads_with_rot_emb(K)   # → [B, L_K, n_heads, head_dim]
        else:
            Qh = self.split_heads(Q)   # → [B, L_Q, n_heads, head_dim]
            Kh = self.split_heads(K)   # → [B, L_K, n_heads, head_dim]
        
        Vh = self.split_heads(V)   # → [B, L_K, n_heads, head_dim]
        Kh, Vh = self.add_zero_token_KhVh(Kh, Vh)

        if self.use_classic_attention:
            out_heads = compute_attention(Qh,Kh,Vh)
        else:
            # 3. Run single‐head linear attention
            # out_heads, _ = self.single_head_attn(
            #     Qh, Kh, Vh, self.g(Qh), self.g(Kh), compute_attn_weight
            # )
            out_heads = fast_linear_path_einsum(Qh,Kh,Vh)
        
        # we can try to use full scaled dot product attention to compare results
        
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


class CrossSelfAttention(nn.Module):
    """
    Module for cross and self attention with time embeddings.
    """
    def __init__(self,in_channels,context_channels,internal_dim=128,use_linear_attention=False):
        """
        in_channels: dim of input channels
        context_channels: dimension of context inputs
        internal_dim: internal dim expansion
        """
        super().__init__()
        def norm(ch):
            # return nn.Identity()
            return nn.RMSNorm(ch)
            # return nn.LayerNorm(ch)
        
        self.input_2_internal = nn.Sequential(
            nn.Linear(in_channels,internal_dim),
            # norm(internal_dim)
        )
        
        self.context_2_internal = nn.Linear(context_channels,internal_dim)
        self.time = nn.Sequential(
            nn.Linear(1,internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim,internal_dim),
        )
        self.context_norm = norm(internal_dim)

        self.sa_QKV =nn.Sequential(
            nn.Linear(
                internal_dim,
                internal_dim*3,
            )
        )
        self.sa_norm = norm(internal_dim)
        self.lsa = MultiHeadLinearAttention(
            internal_dim,
            n_heads=max(4,internal_dim//16),
            dropout=0,
            use_classic_attention = not use_linear_attention,
            add_rotary_emb=True
        )
        self.cross_norm = norm(internal_dim)
        self.lca = MultiHeadLinearAttention(
            internal_dim,
            n_heads=max(4,internal_dim//16),
            dropout=0,
            use_classic_attention = not use_linear_attention
        )
        
        self.cross_Q = nn.Sequential(
            nn.Linear(
                internal_dim,
                internal_dim,
            )
        )
        
        self.cross_KV = nn.Sequential(
            nn.Linear(
                internal_dim,
                internal_dim*2,
            )
        )
        self.mlp_norm = norm(internal_dim)
        self.mlp = Residual([
            nn.Linear(internal_dim,4*internal_dim),
            nn.GELU(),
            nn.Linear(4*internal_dim,in_channels),
        ],init_at_zero=True)
        
    def forward(self,x,context,time):
        """
        x: of shape [BATCH,...dims...,in_channels]
        context: of shape [BATCH,...dims...,context_channels]
        time: of shape [BATCH] with values in range [0;1]
        """
        x_input = x
        x,context = x,context
        x = self.input_2_internal(x)
        context = self.context_2_internal(context)
        context=context+self.time(time)
        
        
        q,k,v = self.sa_QKV(self.sa_norm(x)).chunk(3,-1)
        x = self.lsa(q,k,v)[0]+x
         
        q = self.cross_Q(self.cross_norm(x))
        k,v = self.cross_KV(self.context_norm(context)).chunk(2,-1)
        x = self.lca(q,k,v)[0]+x
        
        return self.mlp(self.mlp_norm(x))+x_input

class EfficientSpatialChannelAttention(nn.Module):
    """
    Efficient Spatial Channel Attention (ESCA) Module

    Applies efficient spatial attention across channels using a grouped [1,2,3]D convolution
    over the spatial dimensions. This module computes attention weights
    that modulate channel responses based on spatial structure, improving
    feature representation with minimal overhead.
    """
    def __init__(self, channels,groups='auto',dimensions=2):
        super().__init__()
        assert dimensions in [1,2,3],f"dimensions must be one of [1,2,3], but got {dimensions}"
        conv = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        
        groups = max(1,channels//32)
        
        self.spatial_attn = nn.Sequential(
            conv(channels,channels,3,padding=1,groups=groups),
            nn.Tanh()
        )

    def forward(self, x):
        spatian_attn = self.spatial_attn(x)
        return x*spatian_attn