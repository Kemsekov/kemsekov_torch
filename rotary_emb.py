from __future__ import annotations
from math import pi, log

import torch
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import List, Literal

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        """
        Parameters:
            dim (int):
                The dimension of the embeddings. This determines the number of frequencies, typically set to half the embedding dimension (dim // 2) for 'lang' and 'pixel' frequency types.
        
            custom_freqs (Tensor | None, default=None):
                Optional custom frequencies to use. If provided, this overrides the frequency calculation based on freqs_for.
            
            freqs_for (Literal['lang', 'pixel', 'constant'], default='lang'):
                Specifies the type of frequencies to compute,\n
                'lang': Computes frequencies optimal for large sequences, commonly used in language models.\n
                'pixel': Uses linearly spaced frequencies from 1 to max_freq / 2, scaled by ππ, suitable for pixel-based inputs.\n
                'constant': Uses a tensor of ones with shape (num_freqs,), for constant frequency embeddings.

            theta (float, default=10000):
                The base value for frequency calculation in 'lang' mode. Adjusted by theta_rescale_factor for sequence length adaptation.
                
            max_freq (float, default=10):
                The maximum frequency for 'pixel' mode, used to define the range of linearly spaced frequencies.
        
            num_freqs (int, default=1):
                The number of frequencies for 'constant' mode.
        
            learned_freq (bool, default=False):
                If True, the frequencies become learnable parameters optimized during training.
            use_xpos (bool, default=False):
                If True, enables the xpos extension, which improves length extrapolation by applying position-dependent scaling.
            
            xpos_scale_base (int, default=512):
                The base value for computing scales in xpos mode.
            
            interpolate_factor (float, default=1.0):
                A factor to scale sequence positions, enabling interpolation for different sequence lengths. Must be >= 1.0.
            
            theta_rescale_factor (float, default=1.0):
                A factor to rescale theta, adapting the embeddings to longer sequences without fine-tuning. Based on NTK-aware scaling.
            
            seq_before_head_dim (bool, default=False):
                If True, assumes the sequence dimension precedes the head dimension in the tensor shape (e.g., (batch, seq, head, dim)). Otherwise, assumes (batch, head, seq, dim).
            
            cache_if_possible (bool, default=True):
                If True, enables caching of computed angles to improve efficiency for repeated computations.
            
            cache_max_seq_len (int, default=8192):
                The maximum sequence length for which angles are cached.
        
        Functionality:
            * Rescales theta using theta_rescale_factor to adjust frequency scaling for longer sequences.
            * Computes or assigns base frequencies based on freqs_for or custom_freqs.
            * Initializes caching buffers (cached_freqs and cached_scales) for angles and scales up to cache_max_seq_len.
            * Registers frequencies as a learnable parameter if learned_freq is True, or as a fixed buffer otherwise.
            * Sets up the default sequence dimension and xpos scaling if enabled.
            * Prepares the module for computing rotary embedding angles.
        """
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible : bool = cache_if_possible
        self.cache_max_seq_len : bool = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len : int = 0

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.factor_out_2 = Rearrange( '... (d r) -> ... d r', r=2)
        self.compress_2_to_out = Rearrange('... d r -> ... (d r)')

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or scale is not None, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return self.apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = self.apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = self.apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            seq_len is not None and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            self.cached_scales is not None and \
            seq_len is not None and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(self, dims : List[int]):
        all_freqs = []
        freqs_per_l = []
        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            freqs = self.forward(pos, seq_len = dim)
            freqs_per_l.append(freqs)
            
        dim = len(dims)-2
        f=freqs_per_l
        
        if dim==0:
            all_freqs = broadcast_tensors(
                f[0][Ellipsis,slice(None, None, None), None,slice(None)],
                f[1][Ellipsis,None, slice(None, None, None),slice(None)]
            )
            
        if dim==1:
            all_freqs = broadcast_tensors(
                f[0][Ellipsis,slice(None, None, None), None, None,slice(None, None, None)],
                f[1][Ellipsis,None, slice(None, None, None), None,slice(None, None, None)],
                f[2][Ellipsis,None, None, slice(None, None, None),slice(None, None, None)]
            )
            
        if dim==2:
            all_freqs = broadcast_tensors(
                f[0][Ellipsis,slice(None, None, None), None, None, None,slice(None)],
                f[1][Ellipsis,None, slice(None, None, None), None, None,slice(None)],
                f[2][Ellipsis,None, None, slice(None, None, None), None,slice(None)],
                f[3][Ellipsis,None, None, None, slice(None, None, None),slice(None)]
            )
            
        if dim==3:
            all_freqs = broadcast_tensors(
                f[0][Ellipsis,slice(None, None, None), None, None, None, None,slice(None)],
                f[1][Ellipsis,None, slice(None, None, None), None, None, None,slice(None)],
                f[2][Ellipsis,None, None, slice(None, None, None), None, None,slice(None)],
                f[3][Ellipsis,None, None, None, slice(None, None, None), None,slice(None)],
                f[4][Ellipsis,None, None, None, None, slice(None, None, None),slice(None)]
            )
        
        return torch.cat(all_freqs, dim = -1)

    def forward(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset : int = 0
    ):
        """
        Description:
        Compute the angles (theta) for rotary embeddings, 
        which are used to apply rotations to query and key vectors in the attention mechanism. 
        These angles are calculated as position * frequency and repeated for sine and cosine components.
        
        Parameters:
            t (Tensor):
                A tensor of sequence positions, typically of shape (seq_len,) or broadcastable to it. Represents the positions for which angles are computed.
            seq_len (int | None, default=None):
                The sequence length. If provided and caching is enabled, checks if angles can be retrieved from the cache.
            offset (int, default=0):
                An offset to apply to the sequence positions, useful for extending beyond cached lengths or handling sliding windows.
        Returns:
            torch.Tensor:
            The computed angles theta with shape (..., dim), where
            For each position in t, angles are computed as t * freqs.
            freqs has length dim // 2, and each angle is repeated twice (for sine and cosine), resulting in dim values per position.
            
        """
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            seq_len is not None and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            self.cached_freqs is not None and \
            seq_len is not None and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.cat([freqs, freqs], dim=-1)

        # freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0 and seq_len is not None:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs

 
    def rotate_half(self,x):
        x = self.factor_out_2(x)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return self.compress_2_to_out(x)

    def apply_rotary_emb(
        self,
        freqs,
        t,
        start_index : int = 0,
        scale: float  = 1.,
        seq_dim: int  = -2,
        freqs_seq_dim : int|None = None
    ):
        dtype = t.dtype

        if freqs_seq_dim is not None:
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0

        # if t.ndim == 3 or freqs_seq_dim is not None:
        #     seq_len = t.shape[seq_dim]
        #     freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)
        
        if t.ndim == 3 or freqs_seq_dim is not None:
            if freqs_seq_dim is not None:
                f_s_dim = int(freqs_seq_dim)
            else:
                f_s_dim=-1
            seq_len = t.shape[seq_dim]
            # Replace slice_at_dim with direct slicing
            if f_s_dim == 0:
                freqs = freqs[-seq_len:, ...]
            elif f_s_dim == 1:
                freqs = freqs[:, -seq_len:]
            else:
                raise ValueError(f"Unsupported freqs_seq_dim: {f_s_dim}")

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        # Split t into three parts: left, middle (to be transformed), and right
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]

        # Apply rotary embeddings without modifying t in place    
        t_transformed = (t_middle * freqs.cos() * scale) + (self.rotate_half(t_middle) * freqs.sin() * scale)
            
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)

        return out.type(dtype)

    # learned rotation helpers

    def apply_learned_rotations(self,rotations, t, start_index = 0, freq_ranges = None):
        if freq_ranges is not None:
            rotations = einsum('..., f -> ... f', rotations, freq_ranges)
            rotations = rearrange(rotations, '... r f -> ... (r f)')

        rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
        return self.apply_rotary_emb(rotations, t, start_index = start_index)

class RotaryEmbInplace(torch.nn.Module):
    """Inplace module that accepts any-dim input x, applies rotary emb and returns it"""
    def __init__(self, in_channels=16,freqs_for : Literal['lang','pixel','constant']='pixel', learned_freq = False):
        """
        Parameters:
            in_channels: input tensor channel dim
            freqs_for : Literal['lang', 'pixel', 'constant'], default='lang'
                Specifies the type of frequencies to compute,

                'lang': Computes frequencies optimal for large sequences, commonly used in language models.

                'pixel': Uses linearly spaced frequencies from 1 to max_freq / 2, scaled by ππ, suitable for pixel-based inputs.

                'constant': Uses a tensor of ones with shape (num_freqs,), for constant frequency embeddings.
            learned_freq: use if you want to learn embedding
        """
        
        super().__init__()
        self.pos_emb = RotaryEmbedding(
            dim = in_channels//4,
            freqs_for = freqs_for,
            max_freq = 256,
            use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
            learned_freq=learned_freq,
        )
        
    def forward(self,x):
        # queries and keys for frequencies to be rotated into
        # say for a video with 8 frames, and rectangular image (feature dimension comes last)

        x_t = x.transpose(1,-1).unsqueeze(1) # batch, heads, dim1,dim2, channels

        # get axial frequencies - (8, 64, 32, 16 * 3 = 48)
        # will automatically do partial rotary

        freqs = self.pos_emb.get_axial_freqs(x_t.shape[1:-1])

        # rotate in frequencies
        x_t_emb = self.pos_emb.apply_rotary_emb(freqs, x_t)

        return x_t_emb[:,0].transpose(1,-1)