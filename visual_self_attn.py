import torch
import torch.nn as nn
from kemsekov_torch.positional_emb import PositionalEncoding
import torch.nn.functional as F

def chunk_2d(input, dims=(-1, -2), chunk_sizes=(8, 8)):
    if input.ndim == 3:  # If input has no batch dimension
        input = input.unsqueeze(0)  # Add batch dimension

    # Split the tensor into chunks along the specified dimensions
    chunks = torch.stack(
        [
            torch.stack(torch.chunk(v, chunk_sizes[1], dims[1]), 1)
            for v in torch.chunk(input, chunk_sizes[0], dims[0])
        ],
        1,
    )
    return chunks


def unchunk_2d(chunks, dims=(-1, -2)):
    # Concatenate along the chunked dimensions to reconstruct the original tensor
    concat_along_inner = torch.cat(
        [torch.cat(list(v.unbind(dim=1)), dim=dims[1]) for v in chunks.unbind(dim=1)], 
        dim=dims[0]
    )
    return concat_along_inner

def unfold_2d(input, patch_size=32):
    """
    Splits the input tensor into non-overlapping patches of the specified size.

    Args:
        input (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (int): Size of the patches to extract.

    Returns:
        torch.Tensor: Tensor containing the patches of shape (B, C, patch_size, patch_size, num_patches_h, num_patches_w).
    """
    B, C, H, W = input.shape

    # Ensure the input dimensions are divisible by the patch size
    assert H % patch_size == 0 and W % patch_size == 0, "Height and Width must be divisible by patch_size"

    # Use unfold to extract patches
    patches = input.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0,3,2,1,4,5)


class VisualMultiheadSelfAttentionFull(nn.Module):
    def __init__(self,in_channels,num_heads=8,patch_size=16,v_q_dim = 512):
        super().__init__()
        self.patch_size=patch_size
        chunk_dim_size = patch_size*patch_size*in_channels
        self.pos_enc = PositionalEncoding(chunk_dim_size)
        self.attn = torch.nn.MultiheadAttention(num_heads=num_heads,embed_dim=chunk_dim_size,vdim=v_q_dim,kdim=v_q_dim,dropout=0.1)
        self.v_q_dim=v_q_dim
        self.Q = nn.Linear(chunk_dim_size,v_q_dim)
        self.K = nn.Linear(chunk_dim_size,v_q_dim)
        self.V = nn.Linear(chunk_dim_size,chunk_dim_size)
        
    def forward(self,x):
        # split image into patches of self.patch_size * self.patch_size
        chunks = unfold_2d(x,patch_size=self.patch_size)
        B,CHX,CHY,C,W,H = chunks.shape
        # flatten out channels and spatial info
        chunks_flat = chunks.flatten(-3)
        # add spatial position encoding
        chunks_flat = chunks_flat+self.pos_enc(chunks_flat)
        
        res_flat = chunks_flat.view(
            B,
            CHX*CHY,
            chunks_flat.shape[-1]
        )
        
        # compute self-attention of input image
        out_flat,b = self.attn(self.V(res_flat),self.Q(res_flat),self.K(res_flat))
        
        # resize output to original chunks
        out = out_flat.view(chunks.shape)
        # get image from chunks
        out = unchunk_2d(out)
        return out
        