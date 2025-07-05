"""
This this module you only ever should use `RotaryEmbHeadsInplace` class!

The current state of this implementation is so that it corresponds to
https://github.com/lucidrains/rotary-embedding-torch?tab=readme-ov-file
when used to get axial rotary embeddings via `RotaryEmbHeadsInplace` and it is fully torch-script compilable.
"""

from __future__ import annotations
from math import pi
import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
from typing import List, Literal

# helper functions
def default(val, d):
    return val if val is not None else d

def take_last(
    t: torch.Tensor,
    seq_len: int,
    dim: int
) -> torch.Tensor:
    # normalize negative dims
    if dim < 0:
        dim = dim + t.ndim
    # compute start index
    total = t.size(dim)
    start = total - seq_len
    return t.narrow(dim, start, seq_len)
# rotary embedding helper functions


def rotate_half(x):
    # x shape: (..., 2*d)
    xs = x.shape
    prefix = list(xs[:-1])
    last = xs[-1]
    
    d = last // 2
    
    # reshape to (..., d, 2)
    x = x.reshape(prefix + [d, 2])
    
    # split along that last dimension
    x1, x2 = x.unbind(dim=-1)       # each is shape (..., d)
    
    # apply the rotation: (-x2, x1)
    x = torch.stack((-x2, x1), dim=-1)  # shape (..., d, 2)
    
    # flatten back to (..., 2*d)
    return x.reshape(prefix + [last])


def apply_rotary_emb(
    freqs,
    t,
    start_index : int = 0,
    scale : float = 1.,
    seq_dim : int = -2,
    freqs_seq_dim : int|None = None
):
    dtype = t.dtype

    if freqs_seq_dim is None:
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or freqs_seq_dim is not None:
        seq_len = t.size(seq_dim)
        fs_dim = freqs_seq_dim if freqs_seq_dim is not None else 0
        freqs = take_last(freqs, seq_len, fs_dim)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


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

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

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

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_axial_freqs(self, dims : List[int], offsets : List[int]|None =None):
        all_freqs = []
        num_dims = len(dims)

        if offsets is None:
            offsets = [0] * num_dims

        assert len(dims) <= 5, "Only supports up to 5 dimensions"

        for ind in range(num_dims):
            dim = dims[ind]
            offset = offsets[ind]

            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            pos = pos + offset
            freqs = self.forward(pos, seq_len=dim)

            # Insert singleton dimensions for correct shape, depending on axis index
            
            # now insert singleton dims via indexing
            if num_dims == 1:
                # want [dim, rot] → [dim, rot] (no change)
                pass

            elif num_dims == 2:
                if ind == 0:
                    # [dim, rot] → [dim, 1, rot]
                    freqs = freqs[:, None, :]
                else:  # ind == 1
                    # [dim, rot] → [1, dim, rot]
                    freqs = freqs[None, :, :]

            elif num_dims == 3:
                if ind == 0:
                    # [dim, rot] → [dim, 1, 1, rot]
                    freqs = freqs[:, None, None, :]
                elif ind == 1:
                    # [dim, rot] → [1, dim, 1, rot]
                    freqs = freqs[None, :, None, :]
                else:  # ind == 2
                    # [dim, rot] → [1, 1, dim, rot]
                    freqs = freqs[None, None, :, :]

            elif num_dims == 4:
                if ind == 0:
                    # [dim, rot] → [dim, 1, 1, 1, rot]
                    freqs = freqs[:, None, None, None, :]
                elif ind == 1:
                    # [dim, rot] → [1, dim, 1, 1, rot]
                    freqs = freqs[None, :, None, None, :]
                elif ind == 2:
                    # [dim, rot] → [1, 1, dim, 1, rot]
                    freqs = freqs[None, None, :, None, :]
                else:  # ind == 3
                    # [dim, rot] → [1, 1, 1, dim, rot]
                    freqs = freqs[None, None, None, :, :]

            else:  # num_dims == 5
                if ind == 0:
                    freqs = freqs[:, None, None, None, None, :]
                elif ind == 1:
                    freqs = freqs[None, :, None, None, None, :]
                elif ind == 2:
                    freqs = freqs[None, None, :, None, None, :]
                elif ind == 3:
                    freqs = freqs[None, None, None, :, None, :]
                else:  # ind == 4
                    freqs = freqs[None, None, None, None, :, :]

            all_freqs.append(freqs)
        if len(all_freqs)==2:
            all_freqs = torch.broadcast_tensors(all_freqs[0],all_freqs[1])
        elif len(all_freqs)==3:
            all_freqs = torch.broadcast_tensors(all_freqs[0],all_freqs[1],all_freqs[2])
        elif len(all_freqs)==4:
            all_freqs = torch.broadcast_tensors(all_freqs[0],all_freqs[1],all_freqs[2],all_freqs[3])
        elif len(all_freqs)==5:
            all_freqs = torch.broadcast_tensors(all_freqs[0],all_freqs[1],all_freqs[2],all_freqs[3],all_freqs[4])
        return torch.cat(all_freqs, dim=-1)

    def forward(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset : int = 0
    ):
        if offset is not None and seq_len is not None:
            offset_p_seqlen=(offset + seq_len)
        else:
            offset_p_seqlen=0
            
        should_cache = (
            self.cache_if_possible and
            seq_len is not None and
            self.freqs_for != 'pixel' and
            offset_p_seqlen <= self.cache_max_seq_len
        )
        if (
            should_cache and \
            self.cached_freqs is not None and \
            offset_p_seqlen <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:offset_p_seqlen].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = freqs.repeat_interleave(repeats=2, dim=-1)

        if should_cache and offset == 0 and seq_len is not None:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs
class RotaryEmbHeadsInplace(torch.nn.Module):
    """
    Inplace module that accepts any-dim input x, applies rotary emb and returns it.
    
    Accepts inputs of shape `(BATCH, HEADS, (...), DIM)`
    where (...) is spatial dimensions
    """
    def __init__(self, in_channels=16,freqs_for : Literal['lang','pixel','constant']='pixel',max_freq=256):
        """
        Inplace module that accepts any-dim input x, applies rotary emb and returns it.
    
        Accepts inputs of shape `(BATCH, HEADS, (...), DIM)` where `(...)` is spatial dimensions
        Parameters:
            in_channels: input tensor channel dim
            freqs_for : Literal['lang', 'pixel', 'constant'], default='lang'
                Specifies the type of frequencies to compute,

                'lang': Computes frequencies optimal for large sequences, commonly used in language models.

                'pixel': Uses linearly spaced frequencies from 1 to max_freq / 2, scaled by ππ, suitable for pixel-based inputs.

                'constant': Uses a tensor of ones with shape (num_freqs,), for constant frequency embeddings.
            max_freq: max amout of frequencies used in rotations
        """
        
        super().__init__()
        self.pos_emb = RotaryEmbedding(
            dim = in_channels//4,
            freqs_for = freqs_for,
            max_freq = max_freq,
            cache_max_seq_len=8192*2
        )
    
    def forward(self,tensors_list : List[torch.Tensor]):
        x_t = tensors_list[0] # batch, heads, dim1,dim2, channels
        freqs = self.pos_emb.get_axial_freqs(x_t.shape[1:-1])
        # rotate in frequencies
        x_t_emb = [apply_rotary_emb(freqs, xt) for xt in tensors_list]
        return x_t_emb
