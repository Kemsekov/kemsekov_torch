# fixed implementation from https://github.com/lucidrains/ITTR-pytorch of paper
# https://arxiv.org/pdf/2203.16015

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Reduce, Rearrange

def l2norm(t):
    return F.normalize(t, dim = 1)

class DPSA(nn.Module):
    def __init__(
        self,
        dim,           # Input channel dimension
        dim_head,      # Output channel dimension
        heads=8,       # Number of attention heads
        top_k=(-1,-1), # top k values that is selected
        dropout=0.0,   # Dropout rate
        ):
        """
        DPSA module that performs double pruned self-attention
        
        **Args:**
            dim: input dimension
            dim_head: dimensions per head
            heads: heads count
            top_k: tuple that defines input dimensions, must be of length 1,2 or 3, specifies how many elements per head to take for attention in each spatial dimension. When left to -1, will use sqrt of input shape.
            dropout: dropout to use
        """
        super().__init__()
        self.dpca = DPCA(dim,dim_head,heads,top_k)
    
    def forward(self,x):
        return self.dpca(x,x)

class DPCA(nn.Module):
    def __init__(
        self,
        dim,           # Input channel dimension
        dim_head,      # Output channel dimension
        heads=8,       # Number of attention heads
        top_k=(-1,-1), # top k values that is selected
        dropout=0.0,   # Dropout rate
        ):
        """
        DPCA module that performs double pruned cross-attention
        
        **Args:**
            dim: input dimension
            dim_head: dimensions per head
            heads: heads count
            top_k: tuple that defines input dimensions, must be of length 1,2 or 3, specifies how many elements per head to take for attention in each spatial dimension. When left to -1, will use sqrt of input shape.
            dropout: dropout to use
        """
        super().__init__()
        assert len(top_k) in [1,2,3],"top_k must be at most of length 3"
        
        dimensions = len(top_k)
        
        if dimensions==1:
            self.DPCA = DPCA1D(
                dim,dim_head,heads,top_k[0],dropout
            )
        
        if dimensions==2:
            self.DPCA = DPCA2D(
                dim,dim_head,heads,top_k[0],top_k[1],dropout
            )
        
        if dimensions==3:
            self.DPCA = DPCA3D(
                dim,dim_head,heads,top_k[0],top_k[1],top_k[2],dropout
            )
    
    def forward(self,context, query_source):
        return self.DPCA(context, query_source)
        

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
        length_top_k=-1,  # Number of sequence positions to keep
        dropout=0.        # Dropout rate
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head  # Per-head dimension
        inner_dim = dim_head * heads  # Total dimension after projection

        # Channel normalization
        self.context_norm = ChanLayerNorm1D(dim)
        self.query_source_norm = ChanLayerNorm1D(dim)

        # Projection to queries, keys, values using a 1x1 convolution
        self.to_kv = nn.Conv1d(dim, inner_dim * 2, kernel_size=1, bias=False)
        self.to_q = nn.Conv1d(dim, inner_dim, kernel_size=1, bias=False)
        
        self.to_out = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False)

        # Pruning parameter
        self.length_top_k = length_top_k

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.zeros(1))
        

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) L -> (b h) c L', h=heads)
        self.q_probe_reduce = Reduce('b c L -> b c', 'sum')
        self.flatten_to_hidden_dim = Rearrange('b d L -> b L d')

    def forward(self, context, query_source):
        #context is used to compute KV
        #condition is used to compute Q
        
        # Input shape: (b, c, L)
        b, c, L = query_source.shape
        length_top_k = self.length_top_k if self.length_top_k>0 else int(L**0.5)
        # Determine if pruning is needed
        need_select = length_top_k < L

        # Normalize input
        context = self.context_norm(context)
        query_source = self.query_source_norm(query_source)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)  # Each: (b, inner_dim, L)
        q = self.to_q(query_source)

        # Fold out heads: (b, inner_dim, L) -> (b * heads, dim_head, L)
        q = self.fold_out_heads(q)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)

        # L2 normalize queries and keys
        q, k = l2norm(q), l2norm(k)

        # Pruning along the sequence length
        if need_select:
            q_abs = torch.abs(q)
            k_abs = torch.abs(k)
            q_probe = self.q_probe_reduce(q_abs)  # (b * heads, dim_head)
            # Compute scores for each position
            score_l = einsum('b c, b c L -> b L', q_probe, k_abs)  # (b * heads, L)
            # Select top-k indices
            top_l_indices = score_l.topk(k=length_top_k, dim=-1).indices  # (b * heads, k_l)
            # Expand indices for gathering
            top_l_indices = top_l_indices[:, None, :].expand(-1, self.dim_head, -1)  # (b * heads, dim_head, k_l)
            # Gather k and v
            k = torch.gather(k, dim=2, index=top_l_indices)
            v = torch.gather(v, dim=2, index=top_l_indices)  # k, v: (b * heads, dim_head, k_l)

        # Flatten spatial dimension
        q = self.flatten_to_hidden_dim(q)  # (b * heads, L, dim_head)
        k = self.flatten_to_hidden_dim(k)  # (b * heads, k_l, dim_head) if pruned, else (b * heads, L, dim_head)
        v = self.flatten_to_hidden_dim(v)

        # Compute attention
        sim = einsum('b i d, b j d -> b i j', q, k)  # (b * heads, L, k_l) or (b * heads, L, L)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)  # (b * heads, L, dim_head)

        # Reshape back to 1D spatial dimension
        out = out.view(b, self.heads, L, self.dim_head)
        out = out.permute(0, 1, 3, 2).contiguous()  # (b, heads, dim_head, L)
        out = out.view(b, self.heads * self.dim_head, L)
        out = self.to_out(out)
        
        # Final projection
        return self.gamma*out+query_source

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
        height_top_k = -1,
        width_top_k = -1,
        dropout = 0.
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head*heads
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.context_norm = ChanLayerNorm2D(dim)
        self.query_source_norm = ChanLayerNorm2D(dim)
        
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False)
        
        self.to_out = nn.Conv2d(inner_dim, dim, kernel_size=1, bias=False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.fold_out_heads = Rearrange('b (h c) ... -> (b h) c ...', h = self.heads)
        self.q_probe_reduce = Reduce('b c ... -> b c', 'sum')
        self.k_sum_over_width = Reduce('b c height width -> b height c', 'sum')
        self.k_sum_over_height = Reduce('b c height width -> b c width', 'sum')
        self.flatten_to_hidden_dim=Rearrange('b d h w -> b (h w) d')

    def forward(self, context, query_source):
        b, c, height,width = query_source.shape
        
        height_top_k = self.height_top_k if self.height_top_k>0 else int(height**0.5)
        width_top_k = self.width_top_k if self.width_top_k>0 else int(width**0.5)
        
        # Normalize input
        context = self.context_norm(context)
        query_source = self.query_source_norm(query_source)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)
        q = self.to_q(query_source)

        q = self.fold_out_heads(q)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)
        
        # fold out heads
        # they used l2 normalized queries and keys, cosine sim attention basically

        q, k = l2norm(q),l2norm(k)

        # calculate whether to select and rank along height and width

        need_height_select_and_rank = height_top_k < height
        need_width_select_and_rank = width_top_k < width

        # select and rank keys / values, probing with query (reduced along height and width) and keys reduced along row and column respectively

        # C is hidden dimension
        
        
        if need_width_select_and_rank or need_height_select_and_rank:
            # use abs for queries to get relative importance
            q_abs = torch.abs(q)
            k_abs = torch.abs(k)
            
            # sum over abs of height and width
            q_probe = self.q_probe_reduce(q_abs)

            # gather along height, then width
            if need_height_select_and_rank:
                # k_abs = torch.abs(k)
                # sum over width
                k_height = self.k_sum_over_width(k_abs)

                score_r = einsum('b c, b h c -> b h', q_probe, k_height)
                top_h_indices = score_r.topk(k = height_top_k, dim = -1).indices
                top_h_indices = top_h_indices[:,None,:,None].expand(-1, k.shape[1], -1, k.shape[-1])
                k, v = torch.gather(k, dim=2, index=top_h_indices),torch.gather(v, dim=2, index=top_h_indices)
            
            if need_width_select_and_rank:
                # k_abs = torch.abs(k)
                # sum over height
                k_width = self.k_sum_over_height(k_abs)

                score_c = einsum('b h, b c w -> b w', q_probe, k_width)
                top_w_indices = score_c.topk(k = width_top_k, dim = -1).indices
                top_w_indices = top_w_indices[:,None,None,:].expand(-1, k.shape[1], k.shape[2], -1)
                k, v = torch.gather(k, dim=3, index=top_w_indices),torch.gather(v, dim=3, index=top_w_indices)
            
        q, k, v = self.flatten_to_hidden_dim(q),self.flatten_to_hidden_dim(k),self.flatten_to_hidden_dim(v)

        # cosine similarities
        sim = einsum('b i d, b j d -> b i j', q, k)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate out
        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge heads and combine out
        out = out.view(b, self.heads, height, width, self.dim_head)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(b, self.heads * self.dim_head, height, width)
        out = self.to_out(out)
        
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
        depth_top_k=-1,   # Number of depth positions to keep
        height_top_k=-1,  # Number of height positions to keep
        width_top_k=-1,   # Number of width positions to keep
        dropout=0.        # Dropout rate
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head  # Per-head dimension
        inner_dim = dim_head * heads  # Total dimension after projection

        # Channel normalization
        self.context_norm = ChanLayerNorm3D(dim)
        self.query_source_norm = ChanLayerNorm3D(dim)

        # Projection to queries, keys, values using a 1x1x1 convolution
        self.to_kv = nn.Conv3d(dim, inner_dim * 2, kernel_size=1, bias=False)
        self.to_q = nn.Conv3d(dim, inner_dim, kernel_size=1, bias=False)
        
        self.to_out = nn.Conv3d(inner_dim, dim, kernel_size=1, bias=False)

        # Pruning parameters
        self.depth_top_k = depth_top_k
        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

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

    def forward(self, context, query_source):
        # Input shape: (b, c, D, H, W)
        b, c, D_query, H_query, W_query = query_source.shape  # query_source dimensions
        _, _, D_context, H_context, W_context = context.shape  # context dimensions
        
        # Set top-k values (using context dimensions where appropriate)
        depth_top_k = self.depth_top_k if self.depth_top_k > 0 else int(D_context ** 0.5)  # e.g., int(16**0.5)=4
        height_top_k = self.height_top_k if self.height_top_k > 0 else int(H_context ** 0.5)  # e.g., int(24**0.5)≈4
        width_top_k = self.width_top_k if self.width_top_k > 0 else int(W_context ** 0.5)  # e.g., int(32**0.5)≈5
        # Normalize input
        context = self.context_norm(context)
        query_source = self.query_source_norm(query_source)

        # Project to queries, keys, values
        k, v = self.to_kv(context).chunk(2, dim=1)
        q = self.to_q(query_source)
        
        # Fold out heads
        q = self.fold_out_heads(q)  # (b * heads, dim_head, D, H, W)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)

        # L2 normalize queries and keys
        q, k = l2norm(q), l2norm(k)
        
        # Determine if pruning is needed
        need_depth_select = depth_top_k < D_context
        need_height_select = height_top_k < H_context
        need_width_select = width_top_k < W_context
        
        # Pruning along depth, height, and width
        if need_depth_select or need_height_select or need_width_select:
            q_abs = torch.abs(q)
            q_probe = self.q_probe_reduce(q_abs)  # (b * heads, dim_head)
            k_abs = torch.abs(k)

            if need_depth_select:
                # k_abs = torch.abs(k)
                k_depth = self.k_sum_over_height_width(k_abs)  # (b * heads, dim_head, D_context)
                score_d = einsum('b c, b c d -> b d', q_probe, k_depth)  # (b * heads, D_context)
                top_d_indices = score_d.topk(k=depth_top_k, dim=-1).indices  # (b * heads, k_d)
                # Fix: Use H_context and W_context instead of H and W
                top_d_indices = top_d_indices[:, None, :, None, None].expand(-1, self.dim_head, -1, H_context, W_context)
                k = torch.gather(k, dim=2, index=top_d_indices)  # (b * heads, dim_head, k_d, H_context, W_context)
                v = torch.gather(v, dim=2, index=top_d_indices)

            if need_height_select:
                # k_abs = torch.abs(k)
                k_height = self.k_sum_over_depth_width(k_abs)  # (b * heads, dim_head, H_context)
                score_h = einsum('b c, b c h -> b h', q_probe, k_height)  # (b * heads, H_context)
                top_h_indices = score_h.topk(k=height_top_k, dim=-1).indices  # (b * heads, k_h)
                top_h_indices = top_h_indices[:, None, None, :, None].expand(-1, self.dim_head, k.shape[2], -1, k.shape[4])
                k = torch.gather(k, dim=3, index=top_h_indices)  # (b * heads, dim_head, k_d, k_h, W_context)
                v = torch.gather(v, dim=3, index=top_h_indices)

            if need_width_select:
                # k_abs = torch.abs(k)
                k_width = self.k_sum_over_depth_height(k_abs)  # (b * heads, dim_head, W_context)
                score_w = einsum('b c, b c w -> b w', q_probe, k_width)  # (b * heads, W_context)
                top_w_indices = score_w.topk(k=width_top_k, dim=-1).indices  # (b * heads, k_w)
                top_w_indices = top_w_indices[:, None, None, None, :].expand(-1, self.dim_head, k.shape[2], k.shape[3], -1)
                k = torch.gather(k, dim=4, index=top_w_indices)  # (b * heads, dim_head, k_d, k_h, k_w)
                v = torch.gather(v, dim=4, index=top_w_indices)
        # Flatten spatial dimensions
        q = self.flatten_to_hidden_dim(q)  # (b * heads, D * H * W, dim_head)
        k = self.flatten_to_hidden_dim(k)  # (b * heads, k_d * k_h * k_w, dim_head)
        v = self.flatten_to_hidden_dim(v)

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
        
        return self.gamma * out + query_source
