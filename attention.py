import time
from typing import List, Literal, Tuple
import einops
import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange
from kemsekov_torch.residual import Residual, ResidualBlock
from kemsekov_torch.common_modules import ChanLayerNorm,get_normalization_from_name

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
        super().__init__()
        self.embed_dim=embed_dim
    
    def forward(self,Q,K,V,phi_Q,phi_K,compute_attn_weight  : bool = False):
        phi_K=phi_K.transpose(-2,-1)
        
        # here we apply RALA-like approach to increase phi_q @ phi_k matrix rank by rescaling each sample
        # by it's relative importance to whole sequence
        q_probe = torch.mean(Q, dim=-2)  # Reduce from (b, L, C) to (b, C) by summing over L
        q_probe/=(self.embed_dim**0.5)
        
        # score_l = torch.einsum("bhc, bhlc -> bhl",q_probe,K).unsqueeze(-2).softmax(-1)
        score_l = (q_probe.unsqueeze(2) @ K.transpose(-2,-1)).softmax(-1)
        phi_K*=score_l
        del q_probe, score_l
        
        # the full version linear attention
        if compute_attn_weight:
            linear_attn = phi_Q @ phi_K
            linear_attn /= linear_attn.sum(-1,keepdim=True)
        else:
            linear_attn = None
        
        # rearanged attention version that have linear complexity
        K_sum = phi_K.sum(-1,keepdim=True)
        KV = phi_K @ V
        linear_out_fast = phi_Q @ KV
        linear_out_fast /= phi_Q @ K_sum + 1e-6
        
        del K_sum,KV

        return linear_out_fast,linear_attn
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
            nn.Tanh()
        )
        
        self.kernel_K = nn.Sequential(
            nn.Linear(embed_dim,embed_dim,device=device),
            nn.Tanh()
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
        
        phi_Q = self.kernel_Q(Q)+1
        phi_K = self.kernel_K(K)+1
        
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

#somewhat usable
def dist_to_random_Q_selection_cosine(Q, K, V, reference_tokens_count: int,top_k: int) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Select keys and values based on cosine similarity to a random subset of queries.
    
    Q: [B, length_q, DIM]
    K, V: [B, length_kv, DIM]
    
    reference_tokens_count: count of reference points that will be used to compute distances
    top_k: how many tokens to select
    
    Returns:
        selected_K, selected_V: [B, top_k, DIM]
    """
    B, length_q, DIM = Q.shape
    length_kv = K.shape[1]

    if top_k >= length_kv:
        return K, V
    K_orig = K
    
    Q = F.normalize(Q, p=2.0, dim=-1)  # [B, top_k, DIM]
    K = F.normalize(K, p=2.0, dim=-1) # [B, length_kv, DIM]
    
    # compute distances
    
    # Generate random indices for Q
    rand_size=(B, min(reference_tokens_count, length_q))
    rand_token_ind = torch.randint(0, length_q, rand_size, device=Q.device)

    # Select Q_small using advanced indexing: [B, top_k, DIM]
    Q_small = Q[torch.arange(B, device=Q.device)[:, None], rand_token_ind, :]

    # Normalize K and Q_small along the last dimension for cosine similarity
    K_norm = K         # [B, length_kv, DIM]
    Q_small_norm = Q_small  # [B, top_k, DIM]

    # Compute cosine similarity: [B, length_kv, top_k]
    # similarity[b, i, j] = cosine_similarity between K[b,i,:] and Q_small[b,j,:]
    cosine_sim = torch.bmm(K_norm, Q_small_norm.transpose(1, 2))  # [B, length_kv, top_k]

    # For each key, find max cosine similarity over Q_small
    max_sim, _ = cosine_sim.max(dim=2)  # [B, length_kv]

    # Select top_k keys with highest max similarity
    _, indices = torch.topk(max_sim, top_k, dim=1, largest=True, sorted=True)  # [B, top_k]

    batch_indices = torch.arange(B, device=K.device)[:, None]

    selected_K = K_orig[batch_indices, indices, :]  # [B, top_k, DIM]
    selected_V = V[batch_indices, indices, :]  # [B, top_k, DIM]

    return selected_K, selected_V

def prune(Q,K,V,top_k : int):
    q_probe = einops.reduce(Q,'b L C -> b C', 'sum')
    k_abs = K#torch.abs(K)
    score_l = torch.einsum('b C, b L C -> b L', q_probe, k_abs)
    top_l_indices = score_l.topk(k=top_k, dim=-1).indices  # (b * heads, length_top_k)
    # Expand indices for gathering
    top_l_indices = top_l_indices[:, :,None].expand(-1, -1, K.shape[-1])  # (b, length_top_k, dim)
    # Gather k
    k_selected = torch.gather(K, dim=1, index=top_l_indices)  # (b * heads, dim_head, length_top_k)
    v_selected = torch.gather(V, dim=1, index=top_l_indices)  # (b * heads, dim_head, length_top_k)
    return k_selected, v_selected

# do not use
def abs_prune(Q,K,V,top_k : int):
    q_probe = einops.reduce(Q,'b L C -> b C', 'sum')
    k_abs = torch.abs(K)
    score_l = torch.einsum('b C, b L C -> b L', q_probe, k_abs)
    top_l_indices = score_l.topk(k=top_k, dim=-1).indices  # (b * heads, length_top_k)
    # Expand indices for gathering
    top_l_indices = top_l_indices[:, :,None].expand(-1, -1, K.shape[-1])  # (b, length_top_k, dim)
    # Gather k
    k_selected = torch.gather(K, dim=1, index=top_l_indices)  # (b * heads, dim_head, length_top_k)
    v_selected = torch.gather(V, dim=1, index=top_l_indices)  # (b * heads, dim_head, length_top_k)
    return k_selected,v_selected

# do not use
def prod_abs_prune(Q,K,V,top_k : int):
    q_probe = einops.reduce(Q,'b L C -> b C', 'sum')
    k_abs = torch.abs(K)*K
    score_l = torch.einsum('b C, b L C -> b L', q_probe, k_abs)
    top_l_indices = score_l.topk(k=top_k, dim=-1).indices  # (b * heads, length_top_k)
    # Expand indices for gathering
    top_l_indices = top_l_indices[:, :,None].expand(-1, -1, K.shape[-1])  # (b, length_top_k, dim)
    # Gather k
    k_selected = torch.gather(K, dim=1, index=top_l_indices)  # (b * heads, dim_head, length_top_k)
    v_selected = torch.gather(V, dim=1, index=top_l_indices)  # (b * heads, dim_head, length_top_k)
    return k_selected,v_selected

# apparently best key,value pruning method
def sum_abs_prune(Q, K, V, top_k : int):
    if top_k>=K.shape[1]:
        return K,V
    
    # K_orig = K
    # Q = F.normalize(Q, p=2.0, dim=-1)  # [B, top_k, DIM]
    # K = F.normalize(K, p=2.0, dim=-1) # [B, length_kv, DIM]
    
    q_probe = torch.sum(Q, dim=1)  # Reduce from (b, L, C) to (b, C) by summing over L
    k_abs = torch.abs(K) + K  # Element-wise absolute value plus original K, shape (b, L, C)
    score_l = torch.sum(q_probe[:, None, :] * k_abs, dim=2)  # Compute scores, shape (b, L)
    top_l_indices = score_l.topk(k=top_k, dim=-1).indices  # Get indices of top-k scores, shape (b, top_k)
    top_l_indices_expanded = top_l_indices[:, :, None].expand(-1, -1, K.shape[-1])  # Expand to (b, top_k, C)
    k_selected = torch.gather(K, dim=1, index=top_l_indices_expanded)  # Gather from K, shape (b, top_k, C)
    v_selected = torch.gather(V, dim=1, index=top_l_indices_expanded)  # Gather from V, shape (b, top_k, C)
    return k_selected, v_selected

class PrunedMultiheadAttention(nn.Module):
    """
    Pruned multihead attention.
    
    Accepts query, key and value as multidimensional tensor with channels.
    
    [BATCH,CHANNELS,DIM1,DIM2,...]
    """
    def __init__(
        self,
        embed_dim : int,
        num_heads : int,
        top_k = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = True,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = True,
        device: int | None = None,
        dtype: int | None = None):
        """
        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
                across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
            top_k: top K elements selection for keys
            dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
            bias: If specified, adds bias to input / output projection layers. Default: ``True``.
            add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``True``.
            add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
                Default: ``True``.
            kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
            vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``True`` (batch, seq, feature).
        Accepts query, key and value as multidimensional tensor with channels.
        
        [BATCH,CHANNELS,DIM1,DIM2,...]
        """
        super().__init__()
        
        self.model=torch.nn.MultiheadAttention(
            embed_dim       =embed_dim,
            num_heads       =num_heads,
            dropout         =dropout,
            bias            =bias,
            add_bias_kv     =add_bias_kv,
            add_zero_attn   =add_zero_attn,
            kdim            =kdim,
            vdim            =vdim,
            batch_first     =batch_first,
            device          =device,
            dtype           =dtype,
        )
        self.num_heads=num_heads
        self.top_k = top_k
        self.extract_heads = Rearrange("B (h C) ... -> (B h) (...) C",h=num_heads)
        self.collect_heads = Rearrange("(B h) L C -> B L (h C)",h=num_heads)
    
    def forward(self,query,keys,values):
        query_shape=query.shape
        
        #--------------------
        # start = time.time()
        # print(query.shape)
        query  = self.extract_heads(query)
        keys   = self.extract_heads(keys)
        values = self.extract_heads(values)
        # print(query.shape)
        
        query = F.normalize(query, p=2.0, dim=-1)  # [B, top_k, DIM]
        keys = F.normalize(keys, p=2.0, dim=-1)
        
        #--------------------
        # print("\textract heads",time.time()-start)
        # start = time.time()
        
        top_k = query.shape[1]//self.num_heads if self.top_k is None else self.top_k
        # for each head to tokens selection
        # keys,values = dist_to_random_Q_selection_cosine(query, keys, values, top_k, top_k)
        keys,values = sum_abs_prune(query, keys, values, top_k)

        #--------------------
        # print("\tselect tokens",time.time()-start)
        # start = time.time()
        
        # concat heads into full dimension
        query  = self.collect_heads(query)
        keys   = self.collect_heads(keys)
        values = self.collect_heads(values)
        
        #--------------------
        # print("\tconcat heads",time.time()-start)
        # start = time.time()
        
        out,_ = self.model.forward(query,keys,values,need_weights=False)
        out = out.permute(0,2,1).view(query_shape)
        
        #--------------------
        # print("\tmultihead-attention",time.time()-start)
        
        return out
