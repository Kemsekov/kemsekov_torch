import torch
import torch.nn as nn
from kemsekov_torch.positional_emb import PositionalEncoding
import torch.nn.functional as F

def chunk_2d(input, dims: tuple[int,int]=(-1, -2), chunk_sizes: tuple[int,int]=(8, 8)):
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


def unchunk_2d(chunks, dims : tuple[int,int]=(-1, -2)):
    # Concatenate along the chunked dimensions to reconstruct the original tensor
    concat_along_inner = torch.cat(
        [torch.cat(list(v.unbind(dim=1)), dim=dims[1]) for v in chunks.unbind(dim=1)], 
        dim=dims[0]
    )
    return concat_along_inner

def unfold_2d(input, patch_size : int =32):
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
    def __init__(self,in_channels,out_channels,num_heads=8,patch_size=16,v_q_dim = 512,dropout_p=0.1):
        super().__init__()
        self.patch_size=patch_size
        self.V_out_channels=out_channels*num_heads
        chunk_dim_size = patch_size*patch_size*self.V_out_channels
        
        self.V_pos_enc = PositionalEncoding(chunk_dim_size)
        self.attn = torch.nn.MultiheadAttention(num_heads=num_heads,embed_dim=chunk_dim_size,vdim=v_q_dim,kdim=v_q_dim,dropout=dropout_p)
        self.v_q_dim=v_q_dim
        
        out_ch = v_q_dim//patch_size//patch_size
        
        self.Q = nn.Sequential(
            nn.Conv2d(in_channels,out_ch,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
        self.K = nn.Sequential(
            nn.Conv2d(in_channels,out_ch,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
        self.V = nn.Sequential(
            nn.Conv2d(in_channels,out_channels*num_heads,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
        self.out_final = nn.Conv2d(out_channels*num_heads,out_channels,3,padding=1)
        
        self.QK_pos_enc = PositionalEncoding(v_q_dim)
        
        self.QBN = nn.BatchNorm1d(v_q_dim)
        self.KBN = nn.BatchNorm1d(v_q_dim)
        self.VBN = nn.BatchNorm1d(chunk_dim_size)
        
        if in_channels!=out_channels:
            self.x_residual = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.x_residual = nn.Identity()
        
    def forward(self,x):
        # split image into patches of self.patch_size * self.patch_size
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        
        x_chunks = unfold_2d(x,patch_size=self.patch_size)
        Q_chunks = unfold_2d(Q,patch_size=self.patch_size).flatten(-3)
        K_chunks = unfold_2d(K,patch_size=self.patch_size).flatten(-3)
        V_chunks = unfold_2d(V,patch_size=self.patch_size).flatten(-3)
        
        chunks_shape = x_chunks.shape
        x_chunks=x_chunks.flatten(-3)
        
        B,CHX,CHY,D = Q_chunks.shape
        # add spatial position encoding

        # QK_pos_enc = self.QK_pos_enc(Q_chunks)
        # V_pos_enc = self.V_pos_enc(V_chunks)
        # Q_chunks = Q_chunks+QK_pos_enc
        # K_chunks = K_chunks+QK_pos_enc
        # V_chunks = V_chunks+V_pos_enc
        
        x_flat = x_chunks.view(B,CHX*CHY,x_chunks.shape[-1])
        Q_flat = Q_chunks.view(B,CHX*CHY,D)
        K_flat = K_chunks.view(B,CHX*CHY,D)
        V_flat = V_chunks.view(B,CHX*CHY,V_chunks.shape[-1])
        # V_flat = self.V(x_flat)
        
        Q_flat = self.QBN(Q_flat.view(-1,Q_flat.shape[-1])).view(Q_flat.shape)
        K_flat = self.KBN(K_flat.view(-1,K_flat.shape[-1])).view(K_flat.shape)
        V_flat = self.VBN(V_flat.view(-1,V_flat.shape[-1])).view(V_flat.shape)
        
        # compute self-attention of input image
        out_flat,b = self.attn(V_flat,Q_flat,K_flat)
        
        chunks_shape = list(chunks_shape)
        chunks_shape[-3]=self.V_out_channels
        # resize output to original chunks
        out = out_flat.view(chunks_shape)
        out_im=unchunk_2d(out)
        
        # make weighted sum of transformed image and original
        # to make it simpler for model to learn at the start
        
        out = self.out_final(out_im)+self.x_residual(x)
        
        return out
        