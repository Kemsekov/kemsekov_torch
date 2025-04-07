# fixed implementation from https://github.com/lucidrains/ITTR-pytorch of paper
# https://arxiv.org/pdf/2203.16015

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Reduce, Rearrange

from kemsekov_torch.residual import ResidualBlock
from kemsekov_torch.positional_emb import ConcatPositionalEmbeddingPermute, AddPositionalEmbeddingPermute

class DPCABlock(torch.nn.Module):
    def __init__(self,dim,heads=8,dimensions=2,dropout=0.0,top_k=-1,normalization='batch'):
        """
        Somewhat optimal cross-attention DPCA block
        
        dim: input dimensions
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        top_k: count of elements to compute per dimension for each token
        """
        dropout_impl = [nn.Dropout1d,nn.Dropout2d,nn.Dropout3d][dimensions-1]
        super().__init__()
        self.dpca = DPCA(dim,dim//heads,heads,dimensions=dimensions,dropout=dropout,top_k=top_k)
        self.mlp = torch.nn.Sequential(
            ResidualBlock(dim,[dim//4,dim],dimensions=dimensions,normalization=normalization),
            dropout_impl(dropout),
            ResidualBlock(dim,[dim//4,dim],dimensions=dimensions,normalization=normalization),
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
    def __init__(self,dim,heads=8,dimensions=2,dropout=0.0,top_k=-1,normalization='batch'):
        """
        Somewhat optimal self-attention DPSA block
        
        dim: input dimensions
        
        heads: heads for attention
        
        dimensions: dimensions count
        
        dropout: dropout to apply to attention layer
        
        top_k: count of elements to compute per dimension for each token
        """
        super().__init__()
        dropout_impl = [nn.Dropout1d,nn.Dropout2d,nn.Dropout3d][dimensions-1]
        self.dpsa = DPSA(dim,dim//heads,heads,dimensions=dimensions,dropout=dropout,top_k=top_k)
        self.mlp = torch.nn.Sequential(
            ResidualBlock(dim,[dim//4,dim],dimensions=dimensions,normalization=normalization),
            dropout_impl(dropout),
            ResidualBlock(dim,[dim//4,dim],dimensions=dimensions,normalization=normalization),
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
        dropout=0.0,   # Dropout rate
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
        dropout=0.0,   # Dropout rate
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
        
from fast_pytorch_kmeans.kmeans import KMeans
def select_best_ind(q,k,v,top_k):
    """
    Selects subset of items from query(q) and keys(k) in such a way, that minimizes difference between full attention and pruned
    """
    if top_k<=0:
        top_k = int((k.shape[0]*k.shape[1])**0.5)
    # q (B,L,C)
    B = q.shape[0]
    q_flat = q.reshape(q.shape[0]*q.shape[1],q.shape[2]) # (B*L,C)
    kmeans = KMeans(top_k)
    q_cluster = kmeans.fit_predict(q_flat).view(q.shape[:-1]) #(B,L)
    # how many times cluster have occurred in each batch of q
    cluster_size = torch.zeros(B, top_k, dtype=torch.long)
    # (B,top_k)
    # contains counts of cluster occurring in each batch of q
    cluster_size.scatter_add_(1, q_cluster, torch.ones_like(q_cluster, dtype=torch.long))

    k_flat = k.reshape(k.shape[0]*k.shape[1],k.shape[2]) # (B*L,C)
    k_cluster_ind = kmeans.predict(k_flat).view(k.shape[:-1])
    
    # k_element_cluster_size have associated count elements in cluster
    # k_element_cluster_size = torch.gather(cluster_size, dim=1, index=k_cluster_ind)

    k_cluster_centers = kmeans.centroids[k_cluster_ind]
    k_dist_to_center = (k_cluster_centers-k).abs().sum(-1)
    
    best_k = k_dist_to_center.argsort(descending=True)[:,:top_k]
    # best_k = k_element_cluster_size.argsort(descending=True)[:,:top_k]
    
    best_k_expanded=best_k.unsqueeze(-1).expand(-1, -1, k.shape[-1])
    k = torch.gather(k, dim=1, index=best_k_expanded)
    v = torch.gather(v, dim=1, index=best_k_expanded)
    return k,v


def l2norm(t):
    return F.normalize(t, dim = 1)

# Channel-wise Layer Normalization for 1D inputs
class ChanLayerNorm1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return self.gamma * (x - mean) / (var.sqrt() + 1e-6) + self.beta

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
        self.dropout = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.zeros(1))
        

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) L -> (b h) c L', h=heads)
        self.q_probe_reduce = Reduce('b c L -> b c', 'sum')
        self.flatten_to_hidden_dim = Rearrange('b d L -> b L d')

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
        k,v = select_best_ind(q,k,v,self.top_k)
        
        # Compute attention
        sim = einsum('b i d, b j d -> b i j', q, k)  # (b * heads, L_query, L_k)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)  # (b * heads, L_query, dim_head)

        # Reshape back to 1D spatial dimension
        out = out.view(b, self.heads, L_query, self.dim_head)
        out = out.permute(0, 1, 3, 2).contiguous()  # (b, heads, dim_head, L_query)
        out = out.view(b, self.heads * self.dim_head, L_query)
        out = self.to_out(out)  # (b, dim, L_query)
        out = self.out_norm(out)
        
        # Final output with residual connection
        return self.gamma * out + query_source

class ChanLayerNorm2D(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

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

        self.dropout = nn.Dropout(dropout)
        self.fold_out_heads = Rearrange('b (h c) ... -> (b h) c ...', h = self.heads)
        self.q_probe_reduce = Reduce('b c ... -> b c', 'sum')
        self.k_sum_over_width = Reduce('b c height width -> b height c', 'sum')
        self.k_sum_over_height = Reduce('b c height width -> b c width', 'sum')
        self.flatten_to_hidden_dim=Rearrange('b d h w -> b (h w) d')

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

        q, k, v = self.flatten_to_hidden_dim(q),self.flatten_to_hidden_dim(k),self.flatten_to_hidden_dim(v)
        k,v = select_best_ind(q,k,v,self.top_k)

        # cosine similarities
        sim = einsum('b i d, b j d -> b i j', q, k)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate out
        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge heads and combine out
        out = out.view(b, self.heads, height_query, width_query, self.dim_head)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(b, self.heads * self.dim_head, height_query, width_query)
        out = self.to_out(out)
        out = self.out_norm(out)
        
        return self.gamma*out+query_source

# Channel-wise Layer Normalization for 3D inputs
class ChanLayerNorm3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return self.gamma * (x - mean) / (var.sqrt() + 1e-6) + self.beta

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
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) d z w -> (b h) c d z w', h=heads)
        self.q_probe_reduce = Reduce('b c d h w -> b c', 'sum')
        self.k_sum_over_height_width = Reduce('b c d h w -> b c d', 'sum')  # For depth scores
        self.k_sum_over_depth_width = Reduce('b c d h w -> b c h', 'sum')   # For height scores
        self.k_sum_over_depth_height = Reduce('b c d h w -> b c w', 'sum')  # For width scores
        self.flatten_to_hidden_dim = Rearrange('b d ... -> b (...) d')

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
        k,v = select_best_ind(q,k,v,self.top_k)
        
        # Compute attention
        sim = einsum('b i d, b j d -> b i j', q, k)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)

        # Reshape back to 3D
        out = out.view(b, self.heads, D_query, H_query, W_query, self.dim_head)
        out = out.permute(0, 1, 5, 2, 3, 4).contiguous()  # (b, heads, dim_head, D, H, W)
        out = out.view(b, self.heads * self.dim_head, D_query, H_query, W_query)
        out = self.to_out(out)
        out = self.out_norm(out)
        
        return self.gamma * out + query_source




