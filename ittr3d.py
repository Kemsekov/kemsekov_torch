import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Reduce, Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim=-1)

# 3D channel layer norm (like the original ChanLayerNorm but for 3d)
class ChanLayerNorm3d(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# a simple residual wrapper
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

##############################################################################
# DPSA3d
##############################################################################

class DPSA3d(nn.Module):
    """Dual-pruned Self-Attention Block for 3d data."""
    def __init__(
        self,
        dim,
        depth_top_k=-1,
        height_top_k=-1,
        width_top_k=-1,
        dim_head=32,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = ChanLayerNorm3d(dim)
        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        
        self.depth_top_k = depth_top_k
        self.height_top_k = height_top_k
        self.width_top_k = width_top_k
        
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)
        
        # Rearrange for multi-head: from (b, inner_dim, D, H, W) to ((b*heads), dim_head, D, H, W)
        self.fold_out_heads = Rearrange('b (i c) d h w -> (b i) c d h w', i=self.heads)
        # Flatten spatial dims: from (b, c, d, h, w) to (b, (d h w), c)
        self.flatten_to_hidden_dim = Rearrange('b c d h w -> b (d h w) c')
        
    def forward(self, x):
        # x: (b, dim, D, H, W)
        b, _, D, H, W = x.shape
        depth_top_k = self.depth_top_k if self.depth_top_k != -1 else int(math.ceil(D ** 0.5))
        height_top_k = self.height_top_k if self.height_top_k != -1 else int(math.ceil(H ** 0.5))
        width_top_k = self.width_top_k if self.width_top_k != -1 else int(math.ceil(W ** 0.5))
        
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Fold out heads
        q = self.fold_out_heads(q)  # (b*heads, dim_head, D, H, W)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)
        
        # Capture original query spatial dimensions (which remain unpruned)
        orig_D, orig_H, orig_W = q.shape[2], q.shape[3], q.shape[4]
        
        # Normalize queries and keys (cosine similarity attention)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # --- Prune keys and values along depth ---
        if depth_top_k < D:
            q_abs = torch.abs(q)
            q_probe_d = q_abs.sum(dim=(3,4))  # (b*heads, dim_head, D)
            k_abs = torch.abs(k)
            k_d = k_abs.sum(dim=(3,4)).permute(0,2,1)  # (b*heads, D, dim_head)
            score_d = torch.einsum('b c d, b d c -> b d', q_probe_d, k_d)
            top_d_indices = score_d.topk(k=depth_top_k, dim=-1).indices  # (b*heads, depth_top_k)
            top_d_indices = top_d_indices.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (b*heads, 1, depth_top_k, 1, 1)
            top_d_indices = top_d_indices.expand(-1, q.shape[1], -1, H, W)
            k = torch.gather(k, dim=2, index=top_d_indices)
            v = torch.gather(v, dim=2, index=top_d_indices)
            D = depth_top_k  # update pruned depth for k and v

        # --- Prune keys and values along height ---
        if height_top_k < H:
            q_abs = torch.abs(q)
            q_probe_h = q_abs.sum(dim=(2,4))  # (b*heads, dim_head, H)
            k_abs = torch.abs(k)
            k_h = k_abs.sum(dim=(2,4)).permute(0,2,1)  # (b*heads, H, dim_head)
            score_h = torch.einsum('b c h, b h c -> b h', q_probe_h, k_h)
            top_h_indices = score_h.topk(k=height_top_k, dim=-1).indices  # (b*heads, height_top_k)
            top_h_indices = top_h_indices.unsqueeze(1).unsqueeze(2).unsqueeze(-1)  # (b*heads, 1, 1, height_top_k, 1)
            top_h_indices = top_h_indices.expand(-1, q.shape[1], D, -1, W)
            k = torch.gather(k, dim=3, index=top_h_indices)
            v = torch.gather(v, dim=3, index=top_h_indices)
            H = height_top_k

        # --- Prune keys and values along width ---
        if width_top_k < W:
            q_abs = torch.abs(q)
            q_probe_w = q_abs.sum(dim=(2,3))  # (b*heads, dim_head, W)
            k_abs = torch.abs(k)
            k_w = k_abs.sum(dim=(2,3)).permute(0,2,1)  # (b*heads, W, dim_head)
            score_w = torch.einsum('b c w, b w c -> b w', q_probe_w, k_w)
            top_w_indices = score_w.topk(k=width_top_k, dim=-1).indices  # (b*heads, width_top_k)
            top_w_indices = top_w_indices.unsqueeze(1).unsqueeze(2).unsqueeze(2)  # (b*heads, 1, 1, 1, width_top_k)
            top_w_indices = top_w_indices.expand(-1, q.shape[1], D, H, -1)
            k = torch.gather(k, dim=4, index=top_w_indices)
            v = torch.gather(v, dim=4, index=top_w_indices)
            W = width_top_k
        
        # Flatten tokens: note that q is full resolution (orig_D*orig_H*orig_W),
        # while k and v are pruned (D*H*W from keys/values)
        q_flat = self.flatten_to_hidden_dim(q)  # (b*heads, orig_D*orig_H*orig_W, dim_head)
        k_flat = self.flatten_to_hidden_dim(k)   # (b*heads, (D*H*W), dim_head)
        v_flat = self.flatten_to_hidden_dim(v)
        
        # Compute attention with cosine similarity
        sim = torch.einsum('b i d, b j d -> b i j', q_flat, k_flat)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out_flat = torch.einsum('b i j, b j d -> b i d', attn, v_flat)
        
        # Reshape output to (b*heads, dim_head, orig_D, orig_H, orig_W)
        out = out_flat.view(q.shape[0], self.dim_head, orig_D, orig_H, orig_W)
        # Merge heads: from (b*heads, dim_head, ...) to (b, heads*dim_head, orig_D, orig_H, orig_W)
        b_heads = out.shape[0]
        b_orig = b_heads // self.heads
        out = out.view(b_orig, self.heads * self.dim_head, orig_D, orig_H, orig_W)
        
        return self.to_out(out)

##############################################################################
# HPB3d
##############################################################################

# Assume get_normalization_from_name is available and returns a callable that constructs a norm layer.
# (For example, get_normalization_from_name(dimensions=3, normalization='instance') might return nn.InstanceNorm3d.)
from kemsekov_torch.residual import get_normalization_from_name

class HPB3d(nn.Module):
    """
    Hybrid Perception Block for 3d data.
    
    This block applies dual-pruned self-attention (via DPSA3d) in parallel with a 
    depthwise 3D convolution. Their outputs are concatenated, projected via a 1x1x1 conv,
    added (as a residual) to the input, and then passed through an FFN.
    """
    
    def __init__(
        self,
        dim,
        dim_head=32,
        heads=8,
        ff_mult=4,
        attn_depth_top_k=-1,
        attn_height_top_k=-1,
        attn_width_top_k=-1,
        attn_dropout=0.,
        ff_dropout=0.,
        normalization='instance'
    ):
        super().__init__()
        self.attn = DPSA3d(
            dim=dim,
            depth_top_k=attn_depth_top_k,
            height_top_k=attn_height_top_k,
            width_top_k=attn_width_top_k,
            heads=heads,
            dim_head=dim_head,
            dropout=attn_dropout
        )
        self.dwconv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.attn_parallel_combine_out = nn.Conv3d(dim * 2, dim, 1)
        
        ff_inner_dim = dim * ff_mult
        norm_layer = get_normalization_from_name(dimensions=3, normalization=normalization)
        self.ff = nn.Sequential(
            nn.Conv3d(dim, ff_inner_dim, 1),
            norm_layer(ff_inner_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            Residual(nn.Sequential(
                nn.Conv3d(ff_inner_dim, ff_inner_dim, 3, padding=1, groups=ff_inner_dim),
                norm_layer(ff_inner_dim),
                nn.GELU(),
                nn.Dropout(ff_dropout)
            )),
            nn.Conv3d(ff_inner_dim, dim, 1),
            norm_layer(dim)
        )

    def forward(self, x):
        # x: (b, dim, D, H, W)
        attn_branch_out = self.attn(x)
        conv_branch_out = self.dwconv(x)
        concatted = torch.cat((attn_branch_out, conv_branch_out), dim=1)
        # Residual addition
        attn_out = self.attn_parallel_combine_out(concatted) + x
        return self.ff(attn_out)
