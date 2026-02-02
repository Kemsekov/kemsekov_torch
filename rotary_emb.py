from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn

class RotEmb(nn.Module):
    """
    (B, (...dims...), Heads, D)
    
    """
    def __init__(self,base : int=10000,scale_interpolation_power:float=1.0):
        """
        base: base frequency used for ROPE
        scale_interpolation_power: powers scale on inference. Common values are 1 or 0.5
        """
        super().__init__()
        dummy_tensor = torch.zeros(1)
        self.freq_cache_1d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor,torch.Tensor]], {
            "_dummy": (dummy_tensor, dummy_tensor)
        })
        self.freq_cache_2d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]], {
            "_dummy": (dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor)
        })
        self.freq_cache_3d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]], {
            "_dummy": (dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor)
        })
        
        self.eval_freq_cache_1d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor,torch.Tensor]], {
            "_dummy": (dummy_tensor, dummy_tensor)
        })
        self.eval_freq_cache_2d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]], {
            "_dummy": (dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor)
        })
        self.eval_freq_cache_3d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]], {
            "_dummy": (dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor)
        })
        
        self.base = base
        self.max_seq_len1d = 1
        self.max_2d_shape = (1,1)
        self.max_3d_shape = (1,1,1)
        self.scale_interpolation_power=float(scale_interpolation_power)
        
    def forward(self,x):
        dims = len(list(x.shape[1:-2]))
        if dims==0: return x
        
        if dims==1:
            x = self.apply_1d_rotary_pos_emb(x)
        elif dims==2:
            x = self.apply_2d_rotary_pos_emb(x)
        elif dims==3:
            x = self.apply_3d_rotary_pos_emb(x)
        else:
            print("Failed to apply rotary emb")    
        return x
    
    def apply_1d_rotary_pos_emb(self,x):
        """
        x: Tensor of shape (batch, seq_len, heads, dim)
        Returns: tensor with RoPE applied to last dim
        """
        dim = x.shape[-1]
        rotate_dim = dim//2*2
        xr = self._apply_rotary_pos_emb(x[...,:rotate_dim])
        return torch.cat([xr,x[...,rotate_dim:]],-1)

    def _apply_rotary_pos_emb(self,x):
        # Shape: (batch, seq_len, heads, dim)
        bsz, seqlen, nheads, dim = x.shape
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE"

        # Compute rotary frequencies
        half_dim = dim // 2
        sin, cos = self.get_1d_freq(x, self.base, seqlen, half_dim)  # (1, seq_len, 1, half_dim)

        # Split the input into even and odd parts (interleaved pairwise rotation)
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return x_rotated

    def get_1d_freq(self, x, base : int, seqlen : int, half_dim : int):
        key = str((seqlen,half_dim))
        
        if not self.training and key in self.eval_freq_cache_1d:
            sin,cos = [v.to(x.device) for v in self.eval_freq_cache_1d[key]]
            return sin,cos
        
        if key in self.freq_cache_1d:
            sin,cos = [v.to(x.device) for v in self.freq_cache_1d[key]]
            return sin,cos
        
        freq_seq = torch.arange(0, half_dim, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (base ** (freq_seq / half_dim))  # shape: (half_dim,)
        
        # Create position indices
        t = torch.arange(seqlen, device=x.device, dtype=torch.float32)  # (seq_len,)
        
        scale = 1.0
        if self.training:
            self.max_seq_len1d = max(self.max_seq_len1d,seqlen)
        else:
            scale=max(1.0,seqlen/self.max_seq_len1d)
            scale=scale**self.scale_interpolation_power
        freqs = torch.einsum("i,j->ij", t, inv_freq/scale)  # (seq_len, half_dim)
        
        # Create sinusoidal embeddings
        sin = torch.sin(freqs)[None, :, None, :]  # (1, seq_len, 1, half_dim)
        cos = torch.cos(freqs)[None, :, None, :]
        if self.training:
            self.freq_cache_1d[key] = sin,cos
        else:
            self.eval_freq_cache_1d[key] = sin,cos
        return sin,cos

    def apply_2d_rotary_pos_emb(self,x):
        """
        Applies 2D rotary positional embeddings (RoPE) along the height and width dimensions of the input tensor.

        This function splits the embedding dimension into four equal parts corresponding to two pairs of rotary embeddings:
        one pair applied along the height axis and one pair along the width axis. The rotary embedding is applied
        using sinusoidal frequencies designed to encode spatial positions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, Heads, D), where
                B = batch size,
                H = height,
                W = width,
                Heads = number of attention heads,
                D = embedding dimension (must be divisible by 4).
            base (int, optional): Base frequency for rotary embeddings. Default is 10000.

        Returns:
            torch.Tensor: Tensor of the same shape as input (B, H, W, Heads, D) with 2D rotary positional embeddings applied.

        Raises:
            AssertionError: If embedding dimension D is not divisible by 4.

        Example:
            >>> x = torch.randn(10, 32, 32, 8, 64)
            >>> out = apply_2d_rotary_pos_emb(x)
            >>> print(out.shape)
            torch.Size([10, 32, 32, 8, 64])
        """
        dim = x.shape[-1]
        rotate_dim = dim//4*4
        xr = self._apply_2d_rotary_pos_emb(x[...,:rotate_dim])
        return torch.cat([xr,x[...,rotate_dim:]],-1)

    def _apply_2d_rotary_pos_emb(self,x):
        B, H, W, nH, D = x.shape
        assert D % 4 == 0
        
        D_half = D // 2
        D_quarter = D // 4

        # D_quarter H and W defines everything
        sin_h, cos_h, sin_w, cos_w = self.get_2d_freqs(x, self.base, H, W, D_quarter)

        # Split input
        x_h1 = x[..., :D_quarter]
        x_h2 = x[..., D_quarter:D_half]
        x_w1 = x[..., D_half:D_half + D_quarter]
        x_w2 = x[..., D_half + D_quarter:]

        # Apply rotary without extra unsqueeze
        h1_rot = x_h1 * cos_h - x_h2 * sin_h
        h2_rot = x_h1 * sin_h + x_h2 * cos_h
        w1_rot = x_w1 * cos_w - x_w2 * sin_w
        w2_rot = x_w1 * sin_w + x_w2 * cos_w

        return torch.cat([h1_rot, h2_rot, w1_rot, w2_rot], dim=-1)

    def get_2d_freqs(self, x, base : int, H : int, W : int, D_quarter : int):
        key = str((D_quarter,H,W))
        if not self.training and key in self.eval_freq_cache_2d:
            sin_h,cos_h,sin_w,cos_w = [v.to(x.device) for v in self.eval_freq_cache_2d[key]]
            return sin_h,cos_h,sin_w,cos_w
        
        if key in self.freq_cache_2d:
            sin_h,cos_h,sin_w,cos_w = [v.to(x.device) for v in self.freq_cache_2d[key]]
            return sin_h,cos_h,sin_w,cos_w
        freq_seq = torch.arange(0, D_quarter, device=x.device).float()
        inv_freq = 1.0 / (base ** (freq_seq / D_quarter))

        h_pos = torch.arange(H, device=x.device).float()
        w_pos = torch.arange(W, device=x.device).float()
        
        scale_h=1.0
        scale_w=1.0
        if self.training:
            self.max_2d_shape=(max(self.max_2d_shape[0],H),max(self.max_2d_shape[1],W))
        else:
            scale_h = max(1.0,H/self.max_2d_shape[0])
            scale_w = max(1.0,W/self.max_2d_shape[1])
            scale_w=scale_w**self.scale_interpolation_power
            scale_h=scale_h**self.scale_interpolation_power
        
        sin_h = torch.sin(torch.einsum("i,j->ij", h_pos, inv_freq/scale_h))  # (H, D/4)
        cos_h = torch.cos(torch.einsum("i,j->ij", h_pos, inv_freq/scale_h))
        sin_w = torch.sin(torch.einsum("i,j->ij", w_pos, inv_freq/scale_w))  # (W, D/4)
        cos_w = torch.cos(torch.einsum("i,j->ij", w_pos, inv_freq/scale_w))

        # Correct shapes for broadcasting: (1, H, 1, 1, D/4) and (1, 1, W, 1, D/4)
        sin_h = sin_h[None, :, None, None, :]  # (1, H, 1, 1, D/4)
        cos_h = cos_h[None, :, None, None, :]
        sin_w = sin_w[None, None, :, None, :]  # (1, 1, W, 1, D/4)
        cos_w = cos_w[None, None, :, None, :]
        if self.training:
            self.freq_cache_2d[key] = sin_h,cos_h,sin_w,cos_w
        else:
            self.eval_freq_cache_2d[key] = sin_h,cos_h,sin_w,cos_w
        return sin_h,cos_h,sin_w,cos_w

    def apply_3d_rotary_pos_emb(self,x):
        """
        x: Tensor of shape (B, H, W, D, Heads, Dim)
        Returns: same shape with rotary applied along 3 axes
        """
        dim = x.shape[-1]
        rotate_dim = dim//6*6
        xr = self._apply_3d_rotary_pos_emb(x[...,:rotate_dim])
        return torch.cat([xr,x[...,rotate_dim:]],-1)

    def _apply_3d_rotary_pos_emb(self,x):
        B, H, W, D, nH, dim = x.shape
        assert dim % 6 == 0, "DIM must be divisible by 6 for 3D RoPE"
        
        d_part = dim // 3
        d_quarter = d_part // 2  # half for each sin/cos pair

        # Generate inverse frequencies
        sin_h, cos_h, sin_w, cos_w, sin_d, cos_d = self.get_3d_freqs(x, self.base, B, H, W, D, d_quarter)

        # Split tensor into 6 quarter-dim slices
        x_h1 = x[..., :d_quarter]
        x_h2 = x[..., d_quarter:d_part]
        x_w1 = x[..., d_part:d_part + d_quarter]
        x_w2 = x[..., d_part + d_quarter:2 * d_part]
        x_d1 = x[..., 2 * d_part:2 * d_part + d_quarter]
        x_d2 = x[..., 2 * d_part + d_quarter:]

        # Apply RoPE for each axis
        x_h1r = x_h1 * cos_h - x_h2 * sin_h
        x_h2r = x_h1 * sin_h + x_h2 * cos_h
        x_w1r = x_w1 * cos_w - x_w2 * sin_w
        x_w2r = x_w1 * sin_w + x_w2 * cos_w
        x_d1r = x_d1 * cos_d - x_d2 * sin_d
        x_d2r = x_d1 * sin_d + x_d2 * cos_d

        # Concatenate back together
        return torch.cat([x_h1r, x_h2r, x_w1r, x_w2r, x_d1r, x_d2r], dim=-1)

    def get_3d_freqs(self, x, base : int, B : int, H : int, W : int, D : int, d_quarter : int):
        key = str((H, W, D, d_quarter))
        if not self.training and key in self.eval_freq_cache_3d:
            sin_h,cos_h,sin_w,cos_w,sin_d,cos_d = [v.to(x.device) for v in self.eval_freq_cache_3d[key]]
            return sin_h,cos_h,sin_w,cos_w,sin_d,cos_d
        
        if key in self.freq_cache_3d:
            sin_h,cos_h,sin_w,cos_w,sin_d,cos_d = [v.to(x.device) for v in self.freq_cache_3d[key]]
            return sin_h,cos_h,sin_w,cos_w,sin_d,cos_d
        
        freq_seq = torch.arange(d_quarter, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freq_seq / d_quarter))

        # Position indices
        h_pos = torch.arange(H, device=x.device, dtype=torch.float32)
        w_pos = torch.arange(W, device=x.device, dtype=torch.float32)
        d_pos = torch.arange(D, device=x.device, dtype=torch.float32)
        
        scale_h=1.0
        scale_w=1.0
        scale_d=1.0
        if self.training:
            self.max_3d_shape=(max(self.max_3d_shape[0],H),max(self.max_3d_shape[1],W),max(self.max_3d_shape[2],D))
        else:
            scale_h = max(1.0,H/self.max_3d_shape[0])
            scale_w = max(1.0,W/self.max_3d_shape[1])
            scale_d = max(1.0,D/self.max_3d_shape[2])
            scale_h=scale_h**self.scale_interpolation_power
            scale_w=scale_w**self.scale_interpolation_power
            scale_d=scale_d**self.scale_interpolation_power
            
        # RoPE sin/cos for each axis
        # *** Note the leading None on sin_h/cos_h so dim0==1 (batch) ***
        sin_h = torch.sin(torch.einsum('i,j->ij', h_pos, inv_freq/scale_h))[None, :, None, None, None, :]
        cos_h = torch.cos(torch.einsum('i,j->ij', h_pos, inv_freq/scale_h))[None, :, None, None, None, :]
        sin_w = torch.sin(torch.einsum('i,j->ij', w_pos, inv_freq/scale_w))[None, None, :, None, None, :]
        cos_w = torch.cos(torch.einsum('i,j->ij', w_pos, inv_freq/scale_w))[None, None, :, None, None, :]
        sin_d = torch.sin(torch.einsum('i,j->ij', d_pos, inv_freq/scale_d))[None, None, None, :, None, :]
        cos_d = torch.cos(torch.einsum('i,j->ij', d_pos, inv_freq/scale_d))[None, None, None, :, None, :]


        if self.training:
            self.freq_cache_3d[key] = sin_h,cos_h,sin_w,cos_w,sin_d,cos_d
        else:
            self.eval_freq_cache_3d[key] = sin_h,cos_h,sin_w,cos_w,sin_d,cos_d
                
        return sin_h,cos_h,sin_w,cos_w,sin_d,cos_d
