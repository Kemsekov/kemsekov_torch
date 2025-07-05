"""
This this module you only ever should use `RotaryEmbHeadsInplace` class!

The current state of this implementation is so that it corresponds to
https://github.com/lucidrains/rotary-embedding-torch?tab=readme-ov-file
when used to get axial rotary embeddings via `RotaryEmbHeadsInplace` and it is fully torch-script compilable.
"""



from __future__ import annotations
from math import pi, log

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, is_tensor, tensor, Tensor

from einops import rearrange, repeat

from typing import List, Literal

# helper functions
def default(val, d):
    return val if val is not None else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

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

# learned rotation helpers

# def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
#     if freq_ranges is not None:
#         rotations = einsum('..., f -> ... f', rotations, freq_ranges)
#         rotations = rearrange(rotations, '... r f -> ... (r f)')

#     rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
#     return apply_rotary_emb(rotations, t, start_index = start_index)

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

    def get_seq_pos(self, seq_len, device, dtype, offset : int = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset : int = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or scale is not None, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset : int = 0):
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

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    # def get_scale(
    #     self,
    #     t: Tensor,
    #     seq_len: int | None = None,
    #     offset : int = 0
    # ):
    #     assert self.use_xpos
    #     if offset is not None and seq_len is not None:
    #         offset_p_seqlen=(offset + seq_len)
    #     else:
    #         offset_p_seqlen=0
    #     should_cache = (
    #         self.cache_if_possible and
    #         seq_len is not None and
    #         offset_p_seqlen <= self.cache_max_seq_len
    #     )

    #     if (
    #         should_cache and \
    #         self.cached_scales is not None and \
    #         (seq_len + offset) <= self.cached_scales_seq_len
    #     ):
    #         return self.cached_scales[offset:offset_p_seqlen]

    #     scale = 1.
    #     if self.use_xpos:
    #         power = (t - len(t) // 2) / self.scale_base
    #         scale = self.scale ** rearrange(power, 'n -> n 1')
    #         scale = repeat(scale, 'n d -> n (d r)', r = 2)

    #     if should_cache and offset == 0:
    #         self.cached_scales[:seq_len] = scale.detach()
    #         self.cached_scales_seq_len = seq_len

    #     return scale

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
            not self.learned_freq and
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
    def __init__(self, in_channels=16,freqs_for : Literal['lang','pixel','constant']='pixel',max_freq=256, learned_freq = False):
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
            learned_freq: use if you want to learn embedding
        """
        
        super().__init__()
        self.pos_emb = RotaryEmbedding(
            dim = in_channels//4,
            freqs_for = freqs_for,
            max_freq = max_freq,
            # set this to True to make rotary embeddings extrapolate 
            # better to sequence lengths greater than the one used 
            # at training time
            use_xpos = True,
            learned_freq=learned_freq,
        )
    def forward(self,tensors_list : List[torch.Tensor]):
        x_t = tensors_list[0] # batch, heads, dim1,dim2, channels
        freqs = self.pos_emb.get_axial_freqs(x_t.shape[1:-1])
        # rotate in frequencies
        x_t_emb = [apply_rotary_emb(freqs, x_t) for xt in tensors_list]
        return x_t_emb
