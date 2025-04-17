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
        self.dpca = DPCA(dim,dim//heads,heads,dropout=dropout,top_k=top_k)
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
        self.dpsa = DPSA(dim,dim//heads,heads,dropout=dropout,top_k=top_k)
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
        top_k=-1, # top k values that is selected
        dropout=0.1,   # Dropout rate
        ):
        """
        DPSA module that performs double pruned multihead self-attention
        
        **Args:**
            dim: input dimension
            dim_head: dimensions per head. I advice you to use it as dim//heads
            heads: heads count
            top_k: specifies how many elements per head to take for attention in each spatial dimension. When left to -1, will use (total tokens)/heads items for attention per each head, so all heads in sum will visit same count of tokens as tokens itself.
            dropout: dropout to use
        """
        super().__init__()
        self.dpca = DPCA(dim,dim_head,heads,top_k,dropout)
    
    def forward(self,x):
        """
        Computes multihead self-attention on given input tensor x.
        """
        return self.dpca(x,x)

def dist_to_random_Q_selection(Q, K, V, top_k : int):
    """
    Performs selection of keys and values based on distance of keys to random subset of queries.\n
    
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
    distances = torch.cdist(K, Q_small, p=1.0)  # [B, some_other_count, top_k]
    
    # Compute minimum distances over Q_small
    min_distances = distances.min(dim=2)[0]  # [B, some_other_count]
    
    # Get indices of top_k smallest minimum distances
    _, indices = torch.topk(min_distances, top_k, dim=1, largest=False, sorted=True)  # [B, top_k]
    
    all_batch_ind = torch.arange(B, device=K.device)[:, None]
    # Select corresponding points from K using advanced indexing
    selected_K = K[all_batch_ind, indices, :]  # [B, top_k, DIM]
    selected_V = V[all_batch_ind, indices, :]  # [B, top_k, DIM]
    
    return selected_K, selected_V

def expand_to_5d_view(tensor):
    new_shape = list(tensor.shape) + [1] * (5 - tensor.dim())
    return tensor.view(new_shape)

class DPCA(nn.Module):
    """ Dual-pruned Cross-attention Block """
    def __init__(
        self,
        dim,              # Input channel dimension
        dim_head,          # Output channel dimension
        heads=8,          # Number of attention heads
        top_k=-1,  # Number of sequence positions to keep
        dropout=0.1,        # Dropout rate
    ):
        """
        DPCA module that performs double pruned multihead cross-attention
        
        **Args:**
            dim: input dimension
            dim_head: dimensions per head. I advice you to use it as dim//heads
            heads: heads count
            top_k: specifies how many elements per head to take for attention in each spatial dimension. When left to -1, will use (total tokens)/heads items for attention per each head, so all heads in sum will visit same count of tokens as tokens itself.
            dropout: dropout to use
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head  # Per-head dimension
        inner_dim = dim_head * heads  # Total dimension after projection

        ln = ChanLayerNorm3D#[ChanLayerNorm1D,ChanLayerNorm2D,ChanLayerNorm3D][dimensions-1]
        
        # Channel normalization
        self.context_norm = ln(dim)
        self.query_source_norm = ln(dim)
        self.out_norm = ln(dim)

        conv = nn.Conv3d#[nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        
        # Projection to queries, keys, values using a 1x1 convolution
        self.to_kv = conv(dim, inner_dim * 2, kernel_size=1, bias=False)
        self.to_q = conv(dim, inner_dim, kernel_size=1, bias=False)
        self.to_out = conv(inner_dim, dim, kernel_size=1, bias=False)
        
        # Pruning parameters
        self.top_k = top_k
        # Dropout for attention weights
        self.dropout = dropout
        self.gamma = nn.Parameter(torch.zeros(1))

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (H c) ... -> (b H) (...) c', H=heads)
        self.flatten_to_hidden_dim = Rearrange('(b H) L c -> b H L c',H=self.heads)
        self.pack_heads = Rearrange('b H L c -> b (H c) L',H=self.heads)

    def forward(self, query_source, context):
        """
        Computes multihead cross attention for given context and query source.
        
        query_source: `(B,C,...dims...)` tensor that is used to compute query(Q) of attention. 
        We need to embed information from context in this tensor. Output shape will be equal to query_source shape.
        
        context: `(B,C,...dims...)` tensor that is used to compute keys(K) and values(V) of attention. It is additional information that we need to embed into query_source.
        
        query_source shape can be != context shape, only batch and channel dimensions needs to match.
        
        When context==query_source, the results will be same as self-attention.
        """
        if len(context.shape)>5 or len(query_source.shape)>5:
            raise RuntimeError("context and query needs to be at most 5-dimensional")
        orig_query_shape = query_source.shape
        
        context = expand_to_5d_view(context)
        query_source = expand_to_5d_view(query_source)
        
        # Normalize inputs
        context = self.context_norm(context)        # (b, c, L_context)
        query_source_n = self.query_source_norm(query_source)  # (b, c, L_query)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)  # Each: (b, inner_dim, L_context)
        q = self.to_q(query_source_n)                 # (b, inner_dim, L_query)

        # Fold out heads: 
        # (b, inner_dim, dim1,dim2 ...) -> (b * heads, dim1*dim2*..., dim_head)
        q = self.fold_out_heads(q)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)
        
        # L2 normalize queries and keys along dim_head dimension
        q = F.normalize(q, dim = -1)
        k = F.normalize(k, dim = -1)
        
        # Determine if pruning is needed based on context sequence length
        L_context = k.shape[1]
        if self.top_k < L_context:
            top_k = self.top_k if self.top_k > 0 else 1+L_context//self.heads
            k,v = dist_to_random_Q_selection(q,k,v,top_k)

        # extract out head dimension
        # (b * heads, L, dim_head) -> (b, heads, L, dim_head)
        q = self.flatten_to_hidden_dim(q)
        k = self.flatten_to_hidden_dim(k)
        v = self.flatten_to_hidden_dim(v)
        
        if self.training:
            dp = self.dropout
        else:
            dp = 0.0
        
        # Compute attention
        # use scale=1.0 because we use l2 normalized vectors anyway
        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=dp,scale=1.0)
        # out of shape (batch,heads,L_query,head_dim)

        out = self.pack_heads(out)
        # out of shape (batch,heads*head_dim,L_query)
        out_shape = list(query_source.shape)
        out_shape[1]=self.heads*self.dim_head
        
        out = out.view(out_shape)
        out = self.to_out(out)  # (b, dim, L_query)
        out = self.out_norm(out)
        
        # Final output with residual connection
        return (self.gamma * out + query_source).view(orig_query_shape)
