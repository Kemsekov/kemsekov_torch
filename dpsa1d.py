import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from torch import einsum

# Helper function for L2 normalization along a specific dimension
def l2norm(t):
    return t / (t.norm(dim=1, keepdim=True) + 1e-6)

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

class DPSA1D(nn.Module):
    def __init__(
        self,
        dim,              # Input channel dimension
        out_dim,          # Output channel dimension
        heads=8,          # Number of attention heads
        length_top_k=-1,  # Number of sequence positions to keep
        dropout=0.        # Dropout rate
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = out_dim  # Per-head dimension
        inner_dim = out_dim * heads  # Total dimension after projection

        # Channel normalization
        self.norm = ChanLayerNorm1D(dim)

        # Projection to queries, keys, values using a 1x1 convolution
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, kernel_size=1, bias=False)

        # Pruning parameter
        self.length_top_k = length_top_k

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.to_out = nn.Conv1d(inner_dim, out_dim, kernel_size=1)

        # Tensor rearrangement utilities
        self.fold_out_heads = Rearrange('b (h c) L -> (b h) c L', h=heads)
        self.q_probe_reduce = Reduce('b c L -> b c', 'sum')
        self.flatten_to_hidden_dim = Rearrange('b d L -> b L d')

    def forward(self, x):
        # Input shape: (b, c, L)
        b, c, L = x.shape
        length_top_k = self.length_top_k if self.length_top_k>0 else int(L**0.5)
        # Determine if pruning is needed
        need_select = length_top_k < L

        # Normalize input
        x = self.norm(x)

        # Project to queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # Each: (b, inner_dim, L)

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

        # Final projection
        return self.to_out(out)

# Example usage
if __name__ == "__main__":
    # Sample 1D input: batch=2, channels=64, sequence length=100
    x = torch.randn(2, 64, 100)
    model = DPSA1D(dim=64, out_dim=64, heads=8, length_top_k=50)
    out = model(x)
    print(out.shape)  # Expected: torch.Size([2, 64, 100])