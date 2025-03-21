# fixed implementation from https://github.com/lucidrains/ITTR-pytorch of paper
# https://arxiv.org/pdf/2203.16015

import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Reduce, Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = 1)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class DPSA(nn.Module):
    """ Dual-pruned Self-attention Block """
    def __init__(
        self,
        dim,
        out_dim,
        heads = 8,
        height_top_k = 16,
        width_top_k = 16,
        dropout = 0.
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = out_dim
        inner_dim = out_dim*heads

        self.norm = ChanLayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv2d(inner_dim, out_dim, 1)
        self.fold_out_heads = Rearrange('b (h c) ... -> (b h) c ...', h = self.heads)
        self.q_probe_reduce = Reduce('b c ... -> b c', 'sum')
        self.k_sum_over_width = Reduce('b c height width -> b height c', 'sum')
        self.k_sum_over_height = Reduce('b c height width -> b c width', 'sum')
        self.flatten_to_hidden_dim=Rearrange('b d h w -> b (h w) d')

    def forward(self, x):
        b, c, height,width = x.shape
        
        height_top_k = self.height_top_k if self.height_top_k>0 else int(height**0.5)
        width_top_k = self.width_top_k if self.width_top_k>0 else int(width**0.5)
        
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        

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
            
            # sum over abs of height and width
            q_probe = self.q_probe_reduce(q_abs)

            # gather along height, then width
            if need_height_select_and_rank:
                k_abs = torch.abs(k)
                # sum over width
                k_height = self.k_sum_over_width(k_abs)

                score_r = torch.einsum('b c, b h c -> b h', q_probe, k_height)
                top_h_indices = score_r.topk(k = height_top_k, dim = -1).indices
                top_h_indices = top_h_indices[:,None,:,None].expand(-1, k.shape[1], -1, k.shape[-1])
                k, v = torch.gather(k, dim=2, index=top_h_indices),torch.gather(v, dim=2, index=top_h_indices)
            
            if need_width_select_and_rank:
                k_abs = torch.abs(k)
                # sum over height
                k_width = self.k_sum_over_height(k_abs)

                score_c = torch.einsum('bh, bcw -> bw', q_probe, k_width)
                top_w_indices = score_c.topk(k = width_top_k, dim = -1).indices
                top_w_indices = top_w_indices[:,None,None,:].expand(-1, k.shape[1], k.shape[2], -1)
                k, v = torch.gather(k, dim=3, index=top_w_indices),torch.gather(v, dim=3, index=top_w_indices)
            
        q, k, v = self.flatten_to_hidden_dim(q),self.flatten_to_hidden_dim(k),self.flatten_to_hidden_dim(v)

        # cosine similarities
        sim = torch.einsum('b i d, b j d -> b i j', q, k)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate out
        out = torch.einsum('b i j, b j d -> b i d', attn, v)

        # merge heads and combine out
        out = out.view(b, self.heads, height, width, self.dim_head)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(b, self.heads * self.dim_head, height, width)
        
        return self.to_out(out)