"""
Fused, CUDA-graph-friendly replacement for `kemsekov_torch.rotary_emb.RotEmb`.

Same public interface as the original (`__init__(base=10000)`, same `forward`
semantics for 1D/2D/3D rotary embeddings, same registered buffers and same
frequency caching behavior). Differences:

- the whole rotate+concat (incl. the tail-copy) is done in ONE triton kernel
  per call (forward), and ONE for the backward — no intermediate buffers, no
  str() dict keys, no `.to()` on the cache-hit path;
- each element is loaded once (the original/inductor path loads it twice and
  does a redundant cat copy of the full tensor);
- backward avoids the zeros-fill + slice_scatter accumulation the eager
  `cat` backward forces.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn
import triton
import triton.language as tl

from kemsekov_torch.rotary_emb import _compute_inv_freq

_BLOCK_R = 16


@triton.jit
def _rope_fwd_kernel(
    x_ptr, out_ptr, s_row, s_c,
    s_e0, s_e1, s_e2,
    cos0_ptr, sin0_ptr, cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr,
    q, rot_dim, D, R, nH, e0, e1, e2,
    A: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_I: tl.constexpr,
):
    """One program = BLOCK_R rows x BLOCK_I pair-indices; each element pair
    (c, c+q) is loaded once and produces both rotated outputs in registers.
    Channels c in [rot_dim, D) are copied (tail). Traffic = 1R + 1W.

    Row positions are derived from the row's memory offset and the actual
    tensor strides, so any positive-stride view works (row-major not
    required)."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    m_r = rows < R
    m_i = i < q
    m_ri = m_r[:, None] & m_i[None, :]

    off64 = rows.to(tl.int64) * s_row
    pos0 = ((off64 // s_e0) % e0).to(tl.int32)
    pos1 = ((off64 // s_e1) % e1).to(tl.int32)
    pos2 = ((off64 // s_e2) % e2).to(tl.int32)

    for a in tl.static_range(A):
        if a == 0:
            cos_ptr, sin_ptr, pos = cos0_ptr, sin0_ptr, pos0
        elif a == 1:
            cos_ptr, sin_ptr, pos = cos1_ptr, sin1_ptr, pos1
        else:
            cos_ptr, sin_ptr, pos = cos2_ptr, sin2_ptr, pos2
        c1 = 2 * a * q + i
        c2 = c1 + q
        offs1 = rows[:, None] * s_row + c1[None, :] * s_c
        offs2 = rows[:, None] * s_row + c2[None, :] * s_c
        x1 = tl.load(x_ptr + offs1, mask=m_ri, other=0.0)
        x2 = tl.load(x_ptr + offs2, mask=m_ri, other=0.0)
        ct = tl.load(cos_ptr + pos[:, None] * q + i[None, :], mask=m_ri, other=0.0)
        st = tl.load(sin_ptr + pos[:, None] * q + i[None, :], mask=m_ri, other=0.0)
        tl.store(out_ptr + offs1, x1 * ct - x2 * st, mask=m_ri)
        tl.store(out_ptr + offs2, x1 * st + x2 * ct, mask=m_ri)

    ct = rot_dim + i
    m_tail = m_r[:, None] & (ct[None, :] < D)
    offs = rows[:, None] * s_row + ct[None, :] * s_c
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m_tail, other=0.0), mask=m_tail)


@triton.jit
def _rope_bwd_kernel(
    do_ptr, dx_ptr, s_row, s_c,
    s_e0, s_e1, s_e2,
    cos0_ptr, sin0_ptr, cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr,
    q, rot_dim, D, R, nH, e0, e1, e2,
    A: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_I: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    m_r = rows < R
    m_i = i < q
    m_ri = m_r[:, None] & m_i[None, :]

    off64 = rows.to(tl.int64) * s_row
    pos0 = ((off64 // s_e0) % e0).to(tl.int32)
    pos1 = ((off64 // s_e1) % e1).to(tl.int32)
    pos2 = ((off64 // s_e2) % e2).to(tl.int32)

    for a in tl.static_range(A):
        if a == 0:
            cos_ptr, sin_ptr, pos = cos0_ptr, sin0_ptr, pos0
        elif a == 1:
            cos_ptr, sin_ptr, pos = cos1_ptr, sin1_ptr, pos1
        else:
            cos_ptr, sin_ptr, pos = cos2_ptr, sin2_ptr, pos2
        c1 = 2 * a * q + i
        c2 = c1 + q
        offs1 = rows[:, None] * s_row + c1[None, :] * s_c
        offs2 = rows[:, None] * s_row + c2[None, :] * s_c
        d1 = tl.load(do_ptr + offs1, mask=m_ri, other=0.0)
        d2 = tl.load(do_ptr + offs2, mask=m_ri, other=0.0)
        ct = tl.load(cos_ptr + pos[:, None] * q + i[None, :], mask=m_ri, other=0.0)
        st = tl.load(sin_ptr + pos[:, None] * q + i[None, :], mask=m_ri, other=0.0)
        tl.store(dx_ptr + offs1, d1 * ct + d2 * st, mask=m_ri)
        tl.store(dx_ptr + offs2, d2 * ct - d1 * st, mask=m_ri)

    ct = rot_dim + i
    m_tail = m_r[:, None] & (ct[None, :] < D)
    offs = rows[:, None] * s_row + ct[None, :] * s_c
    tl.store(dx_ptr + offs, tl.load(do_ptr + offs, mask=m_tail, other=0.0), mask=m_tail)


class _Plan:
    """Precomputed kernel launch plan for one (shape, mode) — the steady-state
    hot path only does: plan lookup -> empty_like -> kernel launch."""
    __slots__ = (
        "A", "q", "rotate_dim", "nH", "e0", "e1", "e2", "bi", "grid", "D",
        "cos0", "sin0", "cos1", "sin1", "cos2", "sin2",
    )


def _is_dense_perm(x):
    """True if x is contiguous or a positive-strided permutation of a dense
    layout (kernel position decomposition is exact for those)."""
    if x.is_contiguous():
        return True
    strides = x.stride()
    if any(s <= 0 for s in strides):
        return False
    dense = 1
    dense_strides = []
    for sz in reversed(x.shape):
        dense_strides.append(dense)
        dense *= sz
    dense_strides.reverse()
    return sorted(strides) == sorted(dense_strides)


class _FusedRotaryApply(torch.autograd.Function):
    """One kernel per direction. x is viewed as (R, D); the row's memory
    offset is decomposed with the tensor's real strides to recover the
    (batch, spatial extents e0/e1/e2, heads) position for the table lookup
    (works for any positive-strided dense layout)."""

    @staticmethod
    def forward(ctx, x, plan):
        D = plan.D
        R = x.numel() // D
        if plan.q == 0:
            return x.clone()

        if not _is_dense_perm(x):
            x = x.contiguous()
        out = torch.empty_like(x)
        s_row = x.stride(-2) if x.dim() >= 2 else 1
        s_c = x.stride(-1)
        s_e0 = x.stride(1) if plan.e0 > 1 else 1
        s_e1 = x.stride(2) if plan.e1 > 1 else 1
        s_e2 = x.stride(3) if plan.e2 > 1 else 1
        _rope_fwd_kernel[plan.grid](
            x, out, s_row, s_c, s_e0, s_e1, s_e2,
            plan.cos0, plan.sin0, plan.cos1, plan.sin1, plan.cos2, plan.sin2,
            plan.q, plan.rotate_dim, D, R, plan.nH, plan.e0, plan.e1, plan.e2,
            A=plan.A, BLOCK_R=_BLOCK_R, BLOCK_I=plan.bi, num_warps=8,
        )
        ctx.save_for_backward(plan.cos0, plan.sin0, plan.cos1, plan.sin1, plan.cos2, plan.sin2)
        ctx.plan = plan
        return out

    @staticmethod
    def backward(ctx, do):
        plan = ctx.plan
        D = plan.D
        R = do.numel() // D
        if plan.q == 0:
            return do.clone(), None

        dx = torch.empty_like(do)
        s_row = do.stride(-2) if do.dim() >= 2 else 1
        s_c = do.stride(-1)
        s_e0 = do.stride(1) if plan.e0 > 1 else 1
        s_e1 = do.stride(2) if plan.e1 > 1 else 1
        s_e2 = do.stride(3) if plan.e2 > 1 else 1
        _rope_bwd_kernel[plan.grid](
            do, dx, s_row, s_c, s_e0, s_e1, s_e2,
            plan.cos0, plan.sin0, plan.cos1, plan.sin1, plan.cos2, plan.sin2,
            plan.q, plan.rotate_dim, D, R, plan.nH, plan.e0, plan.e1, plan.e2,
            A=plan.A, BLOCK_R=_BLOCK_R, BLOCK_I=plan.bi, num_warps=8,
        )
        return dx, None


class FastRotEmb(nn.Module):
    """
    (B, (...dims...), Heads, D)

    Drop-in replacement for kemsekov_torch.rotary_emb.RotEmb (1D/2D/3D RoPE)
    with fused forward/backward kernels.
    """

    def __init__(self, base: int = 10000):
        """
        base: base frequency used for ROPE
        """
        super().__init__()

        dummy_tensor = torch.zeros(1)
        self.freq_cache_1d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor, torch.Tensor]], {})
        self.freq_cache_2d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], {})
        self.freq_cache_3d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], {})
        self.eval_freq_cache_1d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor, torch.Tensor]], {})
        self.eval_freq_cache_2d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], {})
        self.eval_freq_cache_3d = torch.jit.annotate(Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], {})

        self.base = base
        self.register_buffer('max_seq_len1d', torch.tensor([1]))
        self.register_buffer('max_2d_shape', torch.tensor([1, 1]))
        self.register_buffer('max_3d_shape', torch.tensor([1, 1, 1]))

        self._plan_1d = {}
        self._plan_2d = {}
        self._plan_3d = {}

    @staticmethod
    def _make_plan(x, A, rotate_dim, nH, e0, e1, e2, sin0, cos0,
                   sin1=None, cos1=None, sin2=None, cos2=None):
        D = x.shape[-1]
        R = x.numel() // D
        q = rotate_dim // (2 * A)
        if q == 0:
            plan = _Plan()
            plan.q = 0
            plan.D = D
            return plan
        if sin1 is None:
            sin1, cos1 = sin0, cos0
        if sin2 is None:
            sin2, cos2 = sin0, cos0
        bi = max(16, min(256, triton.next_power_of_2(q)))
        grid = (triton.cdiv(R, _BLOCK_R), triton.cdiv(q, bi))
        plan = _Plan()
        plan.A, plan.q, plan.rotate_dim = A, q, rotate_dim
        plan.nH, plan.e0, plan.e1, plan.e2 = nH, e0, e1, e2
        plan.bi, plan.grid, plan.D = bi, grid, D
        plan.cos0, plan.sin0 = cos0, sin0
        plan.cos1, plan.sin1 = cos1, sin1
        plan.cos2, plan.sin2 = cos2, sin2
        return plan

    def forward(self, x):
        dims = len(list(x.shape[1:-2]))
        if dims == 0:
            return x

        if dims == 1:
            x = self.apply_1d_rotary_pos_emb(x)
        elif dims == 2:
            x = self.apply_2d_rotary_pos_emb(x)
        elif dims == 3:
            x = self.apply_3d_rotary_pos_emb(x)
        else:
            print("Failed to apply rotary emb")
        return x

    # ------------------------------------------------------------------ 1D
    def apply_1d_rotary_pos_emb(self, x):
        dim = x.shape[-1]
        rotate_dim = dim // 2 * 2
        return self._apply_rotary_pos_emb(x, rotate_dim)

    def _apply_rotary_pos_emb(self, x, rotate_dim=None):
        # Shape: (batch, seq_len, heads, dim)
        if rotate_dim is None:
            rotate_dim = x.shape[-1]
        bsz, seqlen, nheads, dim = x.shape
        assert rotate_dim % 2 == 0, "Embedding dimension must be even for RoPE"

        plan = self._plan_1d.get((x.shape, self.training))
        if plan is None:
            half_dim = dim // 2
            sin, cos = self.get_1d_freq(x, self.base, seqlen, half_dim)  # (1, seq_len, 1, half_dim)
            sin = sin.reshape(seqlen, half_dim)
            cos = cos.reshape(seqlen, half_dim)
            plan = self._make_plan(x, 1, rotate_dim, nheads, seqlen, 1, 1, sin, cos)
            self._plan_1d[(x.shape, self.training)] = plan
        return _FusedRotaryApply.apply(x, plan)

    def get_1d_freq(self, x, base: int, seqlen: int, half_dim: int):
        key = str((seqlen, half_dim))
        cache = self.eval_freq_cache_1d if not self.training else self.freq_cache_1d

        if key in cache:
            sin, cos = cache[key]
            if sin.device != x.device:
                sin, cos = sin.to(x.device), cos.to(x.device)
                cache[key] = sin, cos
            return sin, cos

        if self.training:
            self.max_seq_len1d[0] = max(self.max_seq_len1d[0], seqlen)
        train_length = self.max_seq_len1d[0]
        inv_freq = _compute_inv_freq(base, half_dim, device=x.device, trained_length=train_length, eval_length=seqlen)

        t = torch.arange(seqlen, device=x.device, dtype=torch.float32)  # (seq_len,)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, half_dim)

        sin = torch.sin(freqs)[None, :, None, :]  # (1, seq_len, 1, half_dim)
        cos = torch.cos(freqs)[None, :, None, :]
        cache[key] = sin, cos
        return sin, cos

    # ------------------------------------------------------------------ 2D
    def apply_2d_rotary_pos_emb(self, x):
        dim = x.shape[-1]
        rotate_dim = dim // 4 * 4
        return self._apply_2d_rotary_pos_emb(x, rotate_dim)

    def _apply_2d_rotary_pos_emb(self, x, rotate_dim=None):
        B, H, W, nH, D = x.shape
        if rotate_dim is None:
            rotate_dim = D
        assert rotate_dim % 4 == 0

        plan = self._plan_2d.get((x.shape, self.training))
        if plan is None:
            D_quarter = D // 4
            sin_h, cos_h, sin_w, cos_w = self.get_2d_freqs(x, self.base, H, W, D_quarter)
            plan = self._make_plan(
                x, 2, rotate_dim, nH, H, W, 1,
                sin_h.reshape(H, D_quarter), cos_h.reshape(H, D_quarter),
                sin_w.reshape(W, D_quarter), cos_w.reshape(W, D_quarter),
            )
            self._plan_2d[(x.shape, self.training)] = plan
        return _FusedRotaryApply.apply(x, plan)

    def get_2d_freqs(self, x, base: int, H: int, W: int, D_quarter: int):
        key = str((D_quarter, H, W))
        cache = self.eval_freq_cache_2d if not self.training else self.freq_cache_2d

        if key in cache:
            out = cache[key]
            if out[0].device != x.device:
                out = tuple(v.to(x.device) for v in out)
                cache[key] = out
            return out

        h_pos = torch.arange(H, device=x.device).float()
        w_pos = torch.arange(W, device=x.device).float()

        if self.training:
            self.max_2d_shape[0] = max(self.max_2d_shape[0], H)
            self.max_2d_shape[1] = max(self.max_2d_shape[1], W)

        inv_freq_h = _compute_inv_freq(base, D_quarter, device=x.device, trained_length=self.max_2d_shape[0], eval_length=H)
        inv_freq_w = _compute_inv_freq(base, D_quarter, device=x.device, trained_length=self.max_2d_shape[1], eval_length=W)

        sin_h = torch.sin(torch.einsum("i,j->ij", h_pos, inv_freq_h))
        cos_h = torch.cos(torch.einsum("i,j->ij", h_pos, inv_freq_h))
        sin_w = torch.sin(torch.einsum("i,j->ij", w_pos, inv_freq_w))
        cos_w = torch.cos(torch.einsum("i,j->ij", w_pos, inv_freq_w))

        sin_h = sin_h[None, :, None, None, :]
        cos_h = cos_h[None, :, None, None, :]
        sin_w = sin_w[None, None, :, None, :]
        cos_w = cos_w[None, None, :, None, :]
        out = (sin_h, cos_h, sin_w, cos_w)
        cache[key] = out
        return out

    # ------------------------------------------------------------------ 3D
    def apply_3d_rotary_pos_emb(self, x):
        dim = x.shape[-1]
        rotate_dim = dim // 6 * 6
        return self._apply_3d_rotary_pos_emb(x, rotate_dim)

    def _apply_3d_rotary_pos_emb(self, x, rotate_dim=None):
        B, H, W, D, nH, dim = x.shape
        if rotate_dim is None:
            rotate_dim = dim
        assert rotate_dim % 6 == 0, "DIM must be divisible by 6 for 3D RoPE"

        plan = self._plan_3d.get((x.shape, self.training))
        if plan is None:
            d_part = dim // 3
            d_quarter = d_part // 2  # half for each sin/cos pair
            sin_h, cos_h, sin_w, cos_w, sin_d, cos_d = self.get_3d_freqs(x, self.base, B, H, W, D, d_quarter)
            plan = self._make_plan(
                x, 3, rotate_dim, nH, H, W, D,
                sin_h.reshape(H, d_quarter), cos_h.reshape(H, d_quarter),
                sin_w.reshape(W, d_quarter), cos_w.reshape(W, d_quarter),
                sin_d.reshape(D, d_quarter), cos_d.reshape(D, d_quarter),
            )
            self._plan_3d[(x.shape, self.training)] = plan
        return _FusedRotaryApply.apply(x, plan)

    def get_3d_freqs(self, x, base: int, B: int, H: int, W: int, D: int, d_quarter: int):
        key = str((H, W, D, d_quarter))
        cache = self.eval_freq_cache_3d if not self.training else self.freq_cache_3d

        if key in cache:
            out = cache[key]
            if out[0].device != x.device:
                out = tuple(v.to(x.device) for v in out)
                cache[key] = out
            return out

        h_pos = torch.arange(H, device=x.device, dtype=torch.float32)
        w_pos = torch.arange(W, device=x.device, dtype=torch.float32)
        d_pos = torch.arange(D, device=x.device, dtype=torch.float32)

        if self.training:
            self.max_3d_shape[0] = max(self.max_3d_shape[0], H)
            self.max_3d_shape[1] = max(self.max_3d_shape[1], W)
            self.max_3d_shape[2] = max(self.max_3d_shape[2], D)

        inv_freq_h = _compute_inv_freq(base, d_quarter, device=x.device, trained_length=self.max_3d_shape[0], eval_length=H)
        inv_freq_w = _compute_inv_freq(base, d_quarter, device=x.device, trained_length=self.max_3d_shape[1], eval_length=W)
        inv_freq_d = _compute_inv_freq(base, d_quarter, device=x.device, trained_length=self.max_3d_shape[2], eval_length=D)

        sin_h = torch.sin(torch.einsum('i,j->ij', h_pos, inv_freq_h))[None, :, None, None, None, :]
        cos_h = torch.cos(torch.einsum('i,j->ij', h_pos, inv_freq_h))[None, :, None, None, None, :]
        sin_w = torch.sin(torch.einsum('i,j->ij', w_pos, inv_freq_w))[None, None, :, None, None, :]
        cos_w = torch.cos(torch.einsum('i,j->ij', w_pos, inv_freq_w))[None, None, :, None, None, :]
        sin_d = torch.sin(torch.einsum('i,j->ij', d_pos, inv_freq_d))[None, None, None, :, None, :]
        cos_d = torch.cos(torch.einsum('i,j->ij', d_pos, inv_freq_d))[None, None, None, :, None, :]

        out = (sin_h, cos_h, sin_w, cos_w, sin_d, cos_d)
        cache[key] = out
        return out
