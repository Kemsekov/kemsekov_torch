import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from torch import einsum

# Helper function for L2 normalization along a specific dimension
def l2norm(t):
    return t / (t.norm(dim=1, keepdim=True) + 1e-6)

# Channel-wise Layer Normalization for 3D inputs
class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return self.gamma * (x - mean) / (var.sqrt() + 1e-6) + self.beta

class DPSA3D(nn.Module):
    def __init__(
        self,
        dim,              # Input channel dimension
        out_dim,          # Output channel dimension
        heads=8,          # Number of attention heads
        depth_top_k=-1,   # Number of depth positions to keep
        height_top_k=-1,  # Number of height positions to keep
        width_top_k=-1,   # Number of width positions to keep
        dropout=0.        # Dropout rate
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = out_dim  # Per-head dimension
        inner_dim = out_dim * heads  # Total dimension after projection

        # Channel normalization
        self.norm = ChanLayerNorm(dim)

        # Projection to queries, keys, values using a 1x1x1 convolution
        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, kernel_size=1, bias=False)

        # Pruning parameters
        self.depth_top_k = depth_top_k
        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.to_out = nn.Conv3d(inner_dim, out_dim, kernel_size=1)

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) d z w -> (b h) c d z w', h=heads)
        self.q_probe_reduce = Reduce('b c d h w -> b c', 'sum')
        self.k_sum_over_height_width = Reduce('b c d h w -> b c d', 'sum')  # For depth scores
        self.k_sum_over_depth_width = Reduce('b c d h w -> b c h', 'sum')   # For height scores
        self.k_sum_over_depth_height = Reduce('b c d h w -> b c w', 'sum')  # For width scores
        self.flatten_to_hidden_dim = Rearrange('b d ... -> b (...) d')

    def forward(self, x):
        # Input shape: (b, c, D, H, W)
        b, c, D, H, W = x.shape
        depth_top_k = self.depth_top_k if self.depth_top_k>0 else int(D**0.5)
        height_top_k = self.height_top_k if self.height_top_k>0 else int(H**0.5)
        width_top_k = self.width_top_k if self.width_top_k>0 else int(W**0.5)
        
        
        # Determine if pruning is needed along each dimension
        need_depth_select = depth_top_k < D
        need_height_select= height_top_k < H
        need_width_select = width_top_k < W

        # Normalize input
        x = self.norm(x)

        # Project to queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # Each: (b, inner_dim, D, H, W)

        # Fold out heads: (b, inner_dim, ...) -> (b * heads, dim_head, ...)
        q = self.fold_out_heads(q)  # (b * heads, dim_head, D, H, W)
        k = self.fold_out_heads(k)
        v = self.fold_out_heads(v)

        # L2 normalize queries and keys
        q, k = l2norm(q), l2norm(k)

        # Pruning along depth, height, and width
        if need_depth_select or need_height_select or need_width_select:
            q_abs = torch.abs(q)
            k_abs = torch.abs(k)
            q_probe = self.q_probe_reduce(q_abs)  # (b * heads, dim_head)

            if need_depth_select:
                k_depth = self.k_sum_over_height_width(k_abs)  # (b * heads, dim_head, D)
                score_d = einsum('b c, b c d -> b d', q_probe, k_depth)  # (b * heads, D)
                top_d_indices = score_d.topk(k=depth_top_k, dim=-1).indices  # (b * heads, k_d)
                top_d_indices = top_d_indices[:, None, :, None, None].expand(-1, self.dim_head, -1, H, W)
                k = torch.gather(k, dim=2, index=top_d_indices)
                v = torch.gather(v, dim=2, index=top_d_indices)  # k, v: (b * heads, dim_head, k_d, H, W)

            if need_height_select:
                k_height = self.k_sum_over_depth_width(k_abs)  # (b * heads, dim_head, H)
                score_h = einsum('b c, b c h -> b h', q_probe, k_height)  # (b * heads, H)
                top_h_indices = score_h.topk(k=height_top_k, dim=-1).indices  # (b * heads, k_h)
                top_h_indices = top_h_indices[:, None, None, :, None]
                k = torch.gather(k, dim=3, index=top_h_indices.expand(-1, self.dim_head, k.shape[2], -1, W))
                v = torch.gather(v, dim=3, index=top_h_indices.expand(-1, self.dim_head, v.shape[2], -1, W))

            if need_width_select:
                k_width = self.k_sum_over_depth_height(k_abs)  # (b * heads, dim_head, W)
                score_w = einsum('b c, b c w -> b w', q_probe, k_width)  # (b * heads, W)
                top_w_indices = score_w.topk(k=width_top_k, dim=-1).indices  # (b * heads, k_w)
                top_w_indices = top_w_indices[:, None, None, None, :]
                k = torch.gather(k, dim=4, index=top_w_indices.expand(-1, self.dim_head, k.shape[2], k.shape[3], -1))
                v = torch.gather(v, dim=4, index=top_w_indices.expand(-1, self.dim_head, v.shape[2], v.shape[3], -1))

        # Flatten spatial dimensions
        q = self.flatten_to_hidden_dim(q)  # (b * heads, D * H * W, dim_head)
        k = self.flatten_to_hidden_dim(k)  # (b * heads, k_d * k_h * k_w, dim_head)
        v = self.flatten_to_hidden_dim(v)

        # Compute attention
        sim = einsum('b i d, b j d -> b i j', q, k)  # (b * heads, D * H * W, k_d * k_h * k_w)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)  # (b * heads, D * H * W, dim_head)

        # Reshape back to 3D spatial dimensions
        out = out.view(b, self.heads, D, H, W, self.dim_head)
        out = out.permute(0, 1, 5, 2, 3, 4).contiguous()  # (b, heads, dim_head, D, H, W)
        out = out.view(b, self.heads * self.dim_head, D, H, W)

        # Final projection
        return self.to_out(out)

# Example usage
if __name__ == "__main__":
    # Sample 3D input: batch=2, channels=64, depth=32, height=32, width=32
    x = torch.randn(2, 64, 32, 32, 32)
    model = DPSA3D(dim=64, out_dim=64, heads=8, depth_top_k=16, height_top_k=16, width_top_k=16)
    out = model(x)
    print(out.shape)  # Expected: torch.Size([2, 64, 32, 32, 32])