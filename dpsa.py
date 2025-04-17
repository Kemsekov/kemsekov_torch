# fixed implementation from https://github.com/lucidrains/ITTR-pytorch of paper
# https://arxiv.org/pdf/2203.16015

import einops
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Reduce, Rearrange
from kemsekov_torch.common_modules import ChanLayerNorm1D,ChanLayerNorm2D,ChanLayerNorm3D
from kemsekov_torch.residual import ResidualBlock

class DPCABlock(torch.nn.Module):
    def __init__(self,dim,mlp_dim,heads=8,dimensions=2,dropout=0.1,top_k=-1,normalization='batch'):
        """
        Somewhat optimal cross-attention DPCA block
        
        dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        top_k: count of elements to compute per dimension for each token
        """
        super().__init__()
        self.dpca = DPCA(dim,dim//heads,heads,dimensions=dimensions,dropout=dropout,top_k=top_k)
        self.mlp = torch.nn.Sequential(
            ResidualBlock(
                dim,
                [mlp_dim,dim],
                dimensions=dimensions,
                normalization=normalization,
                kernel_size=1
            ),
        )
    def forward(self,query_source, context):
        """
        Computes multihead cross attention for given context and query source.
        
        query_source: tensor that is used to compute query(Q) of attention. 
        We need to embed information from context in this tensor. Output shape will be equal to query_source shape.
        
        context: tensor that is used to compute keys(K) and values(V) of attention. It is additional information that we need to embed into query_source.
        
        query_source shape can be != context shape, only batch and channel dimensions needs to match.
        
        When context==query_source, the results will be same as self-attention.
        """
        attn = self.dpca(query_source,context)
        return self.mlp(attn)

class DPSABlock(torch.nn.Module):
    def __init__(self,dim,mlp_dim,heads=8,dimensions=2,dropout=0.1,top_k=-1,normalization='batch'):
        """
        Somewhat optimal self-attention DPSA block
        
        dim: input dimensions
        
        mlp_dim: internal mlp dimension
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        top_k: count of elements to compute per dimension for each token
        """
        super().__init__()
        self.dpsa = DPSA(dim,dim//heads,heads,dimensions=dimensions,dropout=dropout,top_k=top_k)
        self.mlp = torch.nn.Sequential(
            ResidualBlock(
                dim,
                [mlp_dim,dim],
                dimensions=dimensions,
                normalization=normalization,
                kernel_size=1
            ),
        )
    def forward(self,x):
        """
        Computes multihead self-attention for given input x.
        """
        attn = self.dpsa(x)
        return self.mlp(attn)

class DPSA(nn.Module):
    def __init__(
        self,
        dim,           # Input channel dimension
        dim_head,      # Output channel dimension
        heads=8,       # Number of attention heads
        dimensions=2,
        top_k=-1, # top k values that is selected
        dropout=0.1,   # Dropout rate
        ):
        """
        DPSA module that performs double pruned multihead self-attention
        
        **Args:**
            dim: input dimension
            dim_head: dimensions per head. I advice you to use it as dim//heads
            heads: heads count
            top_k: specifies how many elements per head to take for attention in each spatial dimension. When left to -1, will use sqrt of input dimensions shape. When set to infinity, computes ordinary cross attention
            dropout: dropout to use
        """
        super().__init__()
        self.dpca = DPCA(dim,dim_head,heads,dimensions,top_k,dropout)
    
    def forward(self,x):
        """
        Computes multihead self-attention on given input tensor x.
        """
        return self.dpca(x,x)

class DPCA(nn.Module):
    def __init__(
        self,
        dim,           # Input channel dimension
        dim_head,      # Output channel dimension
        heads=8,       # Number of attention heads
        dimensions=2,
        top_k=-1, # top k values that is selected
        dropout=0.1,   # Dropout rate
        ):
        """
        DPCA module that performs double pruned multihead cross-attention
        
        **Args:**
            dim: input dimension
            dim_head: dimensions per head. I advice you to use it as dim//heads
            heads: heads count
            dimensions: input shapes spatial dimensions
            top_k: specifies how many elements per head to take for attention in each spatial dimension. When left to -1, will use sqrt of input dimensions shape. When set to infinity, computes ordinary cross attention
            dropout: dropout to use
        """
        super().__init__()
        assert dimensions in [1,2,3],"top_k must be at most of length 3"
        
        if dimensions==1:
            self.DPCA = DPCA1D(
                dim,dim_head,heads,top_k,dropout
            )
        
        if dimensions==2:
            self.DPCA = DPCA2D(
                dim,dim_head,heads,top_k,dropout
            )
        
        if dimensions==3:
            self.DPCA = DPCA3D(
                dim,dim_head,heads,top_k,dropout
            )
    
    def forward(self,query_source, context):
        """
        Computes multihead cross attention for given context and query source.
        
        query_source: tensor that is used to compute query(Q) of attention. 
        We need to embed information from context in this tensor. Output shape will be equal to query_source shape.
        
        context: tensor that is used to compute keys(K) and values(V) of attention. It is additional information that we need to embed into query_source.
        
        query_source shape can be != context shape, only batch and channel dimensions needs to match.
        
        When context==query_source, the results will be same as self-attention.
        """
        return self.DPCA(query_source, context)

def l2norm(t):
    return F.normalize(t, dim = 1)

def dist_to_random_Q_selection(Q, K, V, top_k):
    """
    Q of shape [batch,length_q,dim] \n
    K,V of shape [batch,length_kv,dim]
    """
    if top_k>=K.shape[1]:
        return K,V
    B, tokens_count, DIM = Q.shape
    # Generate random indices for Q
    rand_token_ind = torch.randint(0, tokens_count, (B, min(top_k,tokens_count)), device=Q.device)
    
    # Select Q_small using advanced indexing
    Q_small = Q[torch.arange(B, device=Q.device)[:, None], rand_token_ind, :]  # [B, top_k, DIM]
    
    # Compute L1 distances using torch.cdist
    distances = torch.cdist(K, Q_small, p=1)  # [B, some_other_count, top_k]
    
    # Compute minimum distances over Q_small
    min_distances = distances.min(dim=2)[0]  # [B, some_other_count]
    
    # Get indices of top_k smallest minimum distances
    _, indices = torch.topk(min_distances, top_k, dim=1, largest=False, sorted=True)  # [B, top_k]
    
    all_batch_ind = torch.arange(B, device=K.device)[:, None]
    # Select corresponding points from K using advanced indexing
    selected_K = K[all_batch_ind, indices, :]  # [B, top_k, DIM]
    selected_V = V[all_batch_ind, indices, :]  # [B, top_k, DIM]
    
    return selected_K, selected_V
def dist_to_random_Q_selection_with_heads(Q, K, V, top_k):
    """
    Wrapper function for Q, K, V of shape [batch, heads, length, dim].
    Q of shape [batch, heads, length_q, dim]
    K, V of shape [batch, heads, length_kv, dim]
    Returns selected_K, selected_V of shape [batch, heads, top_k, dim]
    """
    B, heads, length_q, dim = Q.shape
    length_kv = K.shape[2]
    
    # Reshape inputs to [B*heads, length, dim]
    Q_reshaped = Q.view(B * heads, length_q, dim)
    K_reshaped = K.view(B * heads, length_kv, dim)
    V_reshaped = V.view(B * heads, length_kv, dim)
    
    # Call original function
    selected_K, selected_V = dist_to_random_Q_selection(Q_reshaped, K_reshaped, V_reshaped, top_k)
    
    # Reshape outputs to [B, heads, top_k, dim]
    selected_K = selected_K.view(B, heads, top_k, dim)
    selected_V = selected_V.view(B, heads, top_k, dim)
    
    return selected_K, selected_V

class DPCA1D(nn.Module):
    """ Dual-pruned Cross-attention Block """
    def __init__(
        self,
        dim,              # Input channel dimension
        dim_head,          # Output channel dimension
        heads=8,          # Number of attention heads
        top_k=-1,  # Number of sequence positions to keep
        dropout=0.        # Dropout rate
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head  # Per-head dimension
        inner_dim = dim_head * heads  # Total dimension after projection

        # Channel normalization
        self.context_norm = ChanLayerNorm1D(dim)
        self.query_source_norm = ChanLayerNorm1D(dim)
        self.out_norm = ChanLayerNorm1D(dim)

        # Projection to queries, keys, values using a 1x1 convolution
        self.to_kv = nn.Conv1d(dim, inner_dim * 2, kernel_size=1, bias=False)
        self.to_q = nn.Conv1d(dim, inner_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False)

        # Pruning parameter
        self.top_k = top_k

        # Dropout for attention weights
        self.dropout = dropout
        self.gamma = nn.Parameter(torch.zeros(1))

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) L -> (b h) c L', h=heads)
        self.flatten_to_hidden_dim = Rearrange('(b H) d L -> b H L d',H=self.heads)

    def forward(self, query_source, context):
        """
        Args:
            context: Tensor of shape (b, c, L_context) - source of keys and values
            query_source: Tensor of shape (b, c, L_query) - source of queries
        Returns:
            Tensor of shape (b, dim, L_query)
        """
        # Unpack shapes separately
        b, c, L_query = query_source.shape

        # Normalize inputs
        context = self.context_norm(context)        # (b, c, L_context)
        query_source_n = self.query_source_norm(query_source)  # (b, c, L_query)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)  # Each: (b, inner_dim, L_context)
        q = self.to_q(query_source_n)                 # (b, inner_dim, L_query)

        # Fold out heads: (b, inner_dim, L) -> (b * heads, dim_head, L)
        q = self.fold_out_heads(q)  # (b * heads, dim_head, L_query)
        k = self.fold_out_heads(k)  # (b * heads, dim_head, L_context)
        v = self.fold_out_heads(v)  # (b * heads, dim_head, L_context)

        # L2 normalize queries and keys
        q, k = l2norm(q), l2norm(k)

        q = self.flatten_to_hidden_dim(q)  # (b * heads, L_query, dim_head)
        k = self.flatten_to_hidden_dim(k)  # (b * heads, L_k, dim_head), L_k = length_top_k if pruned, else L_context
        v = self.flatten_to_hidden_dim(v)  # Same as k
        
        # Determine if pruning is needed based on context sequence length
        L_context = k.shape[2]
        if self.top_k < L_context:
            top_k = self.top_k if self.top_k > 0 else int(L_context // self.heads)
            k,v = dist_to_random_Q_selection_with_heads(q,k,v,top_k)
        if self.training:
            dp = self.dropout
        else:
            dp = 0.0
        # Compute attention
        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=dp,scale=1.0)
        
        # Reshape back to 1D spatial dimension
        out = out.permute(0, 1, 3, 2).contiguous()  # (b, heads, dim_head, L_query)
        out = out.view(b, self.heads * self.dim_head, L_query)
        out = self.to_out(out)  # (b, dim, L_query)
        out = self.out_norm(out)
        
        # Final output with residual connection
        return self.gamma * out + query_source

class DPCA2D(nn.Module):
    """ Dual-pruned Cross-attention Block """
    def __init__(
        self,
        dim,
        dim_head,
        heads = 8,
        top_k = -1,
        dropout = 0.
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head*heads
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.context_norm = ChanLayerNorm2D(dim)
        self.query_source_norm = ChanLayerNorm2D(dim)
        self.out_norm = ChanLayerNorm2D(dim)
        
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, kernel_size=1, bias=False)

        self.top_k = top_k

        self.dropout = dropout
        self.fold_out_heads = Rearrange('b (h c) ... -> (b h) c ...', h = self.heads)
        self.flatten_to_hidden_dim=Rearrange('(b H) d h w -> b H (h w) d',H = self.heads)

    def forward(self, query_source, context):
        # Unpack shapes for query_source and context separately
        b, c, height_query, width_query = query_source.shape

        
        # Normalize input
        context = self.context_norm(context)
        query_source_n = self.query_source_norm(query_source)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)
        q = self.to_q(query_source_n)

        q = self.fold_out_heads(q)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)
        
        # fold out heads
        # they used l2 normalized queries and keys, cosine sim attention basically

        q, k = l2norm(q),l2norm(k)

        q=self.flatten_to_hidden_dim(q)
        k=self.flatten_to_hidden_dim(k)
        v=self.flatten_to_hidden_dim(v)
        
        # Determine if pruning is needed based on context sequence length
        L_context = k.shape[2]
        if self.top_k < L_context:
            top_k = self.top_k if self.top_k > 0 else int(L_context // self.heads)
            k,v = dist_to_random_Q_selection_with_heads(q,k,v,top_k)

        if self.training:
            dp = self.dropout
        else:
            dp = 0.0
        # Compute attention
        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=dp,scale=1.0)
        # merge heads and combine out
        out = out.view(b, self.heads, height_query, width_query, self.dim_head)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(b, self.heads * self.dim_head, height_query, width_query)
        out = self.to_out(out)
        out = self.out_norm(out)
        
        return self.gamma*out+query_source


class DPCA3D(nn.Module):
    """ Dual-pruned Cross-attention Block """
    def __init__(
        self,
        dim,              # Input channel dimension
        dim_head,          # Output channel dimension
        heads=8,          # Number of attention heads
        top_k=-1,   # Number of depth positions to keep
        dropout=0.        # Dropout rate
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head  # Per-head dimension
        inner_dim = dim_head * heads  # Total dimension after projection

        # Channel normalization
        self.context_norm = ChanLayerNorm3D(dim)
        self.query_source_norm = ChanLayerNorm3D(dim)
        self.out_norm = ChanLayerNorm3D(dim)

        # Projection to queries, keys, values using a 1x1x1 convolution
        self.to_kv = nn.Conv3d(dim, inner_dim * 2, kernel_size=1, bias=False)
        self.to_q = nn.Conv3d(dim, inner_dim, kernel_size=1, bias=False)
        
        self.to_out = nn.Conv3d(inner_dim, dim, kernel_size=1, bias=False)

        # Pruning parameters
        self.top_k = top_k

        # Dropout for attention weights
        self.dropout = dropout
        self.gamma = nn.Parameter(torch.zeros(1))

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) d z w -> (b h) c d z w', h=heads)
        self.flatten_to_hidden_dim = Rearrange('(b H) d ... -> b H (...) d',H=self.heads)

    def forward(self, query_source, context):
        # Input shape: (b, c, D, H, W)
        b, c, D_query, H_query, W_query = query_source.shape  # query_source dimensions

        # Normalize input
        context = self.context_norm(context)
        query_source_n = self.query_source_norm(query_source)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)
        q = self.to_q(query_source_n)
        
        # Fold out heads
        q = self.fold_out_heads(q)  # (b * heads, dim_head, D, H, W)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)

        # L2 normalize queries and keys
        q, k = l2norm(q), l2norm(k)
        
        # Flatten spatial dimensions
        q = self.flatten_to_hidden_dim(q)  # (b * heads, D * H * W, dim_head)
        k = self.flatten_to_hidden_dim(k)  # (b * heads, k_d * k_h * k_w, dim_head)
        v = self.flatten_to_hidden_dim(v)

        # Determine if pruning is needed based on context sequence length
        L_context = k.shape[2]
        if self.top_k < L_context:
            top_k = self.top_k if self.top_k > 0 else int(L_context // self.heads)
            k,v = dist_to_random_Q_selection_with_heads(q,k,v,top_k)
        
        if self.training:
            dp = self.dropout
        else:
            dp = 0.0
        # Compute attention
        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=dp,scale=1.0)

        # Reshape back to 3D
        out = out.view(b, self.heads, D_query, H_query, W_query, self.dim_head)
        out = out.permute(0, 1, 5, 2, 3, 4).contiguous()  # (b, heads, dim_head, D, H, W)
        out = out.view(b, self.heads * self.dim_head, D_query, H_query, W_query)
        out = self.to_out(out)
        out = self.out_norm(out)
        
        return self.gamma * out + query_source




