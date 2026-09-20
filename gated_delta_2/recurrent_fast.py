"""Optimized fused recurrent Triton kernels for the Gated Delta Rule-2.

The reference ``recurrent.py`` splits the value dimension across programs and
materialises one gradient buffer per value tile, then reduces them with four
separate ``sum(dim=1)`` kernels.  This module instead accumulates the key-side
gradients directly into ``(batch, L, DK)`` buffers with relaxed atomic adds, so
the backward pass needs no per-tile buffers and no reduction kernels (at
``DV=64`` that removes 4 buffers of ``4 * L * DK`` bytes each per row).

The forward kernel is the exact same per-token recurrence as the serial
reference; the state lives in registers and is checkpointed every
``BLOCK_T`` tokens for the backward pass.
"""

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU-only torch without triton
    _HAS_TRITON = False

_BLOCK_T = 8
_MAX_DIM = 256
# above this many bytes of partial-gradient buffers the backward switches from
# per-tile buffers + reduction kernels to direct atomic accumulation
_ATOMIC_BYTES = int(float(os.environ.get("GD2_ATOMIC_MB", "64")) * 2**20)

# (BLOCK_DV, num_warps) tuned per (DK, DV) bucket
_TUNE = {
    (64, 64): (8, 1),
    (128, 128): (8, 1),
    (128, 64): (8, 1),
    (64, 128): (8, 1),
}
_TUNE_DEFAULT = (8, 1)


if _HAS_TRITON:

    @triton.jit
    def _gdr_fwd(
        A, K, E, Q, Z, O, SB,
        L,
        stride_b, stride_bv,
        DK: tl.constexpr, DV: tl.constexpr,
        BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        NV = tl.cdiv(DV, BLOCK_DV)
        i_v = pid % NV
        bid = pid // NV
        rk = tl.arange(0, BLOCK_DK)
        rv = tl.arange(0, BLOCK_DV)
        mk = rk < DK
        mv = i_v * BLOCK_DV + rv < DV
        a_p = A + bid * stride_b
        k_p = K + bid * stride_b
        e_p = E + bid * stride_b
        q_p = Q + bid * stride_b
        z_p = Z + bid * stride_bv + i_v * BLOCK_DV
        o_p = O + bid * stride_bv + i_v * BLOCK_DV
        nblk = (L + BLOCK_T - 1) // BLOCK_T
        sb_p = SB + (bid * nblk) * DV * DK + i_v * BLOCK_DV * DK
        m2d = mv[:, None] & mk[None, :]
        H = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
        for t in range(L):
            if t % BLOCK_T == 0:
                tl.store(
                    sb_p + (t // BLOCK_T) * DV * DK + rv[:, None] * DK
                    + rk[None, :],
                    H, mask=m2d,
                )
            a_t = tl.load(a_p + t * DK + rk, mask=mk, other=1.0)
            k_t = tl.load(k_p + t * DK + rk, mask=mk, other=0.0)
            e_t = tl.load(e_p + t * DK + rk, mask=mk, other=0.0)
            q_t = tl.load(q_p + t * DK + rk, mask=mk, other=0.0)
            z_t = tl.load(z_p + t * DV + rv, mask=mv, other=0.0)
            H = a_t[None, :] * H
            diff = z_t - tl.sum(H * e_t[None, :], axis=1)
            H = H + diff[:, None] * k_t[None, :]
            o = tl.sum(H * q_t[None, :], axis=1)
            tl.store(o_p + t * DV + rv, o, mask=mv)

    @triton.jit
    def _gdr_bwd(
        A, K, E, Q, Z, GO, DA, DGK, DE, DQ, DZ, SB,
        L,
        stride_b, stride_bv,
        DK: tl.constexpr, DV: tl.constexpr,
        BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
        BLOCK_T: tl.constexpr, ATOMIC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        NV = tl.cdiv(DV, BLOCK_DV)
        i_v = pid % NV
        bid = pid // NV
        rk = tl.arange(0, BLOCK_DK)
        rv = tl.arange(0, BLOCK_DV)
        mk = rk < DK
        mv = i_v * BLOCK_DV + rv < DV
        a_p = A + bid * stride_b
        k_p = K + bid * stride_b
        e_p = E + bid * stride_b
        q_p = Q + bid * stride_b
        z_p = Z + bid * stride_bv + i_v * BLOCK_DV
        go_p = GO + bid * stride_bv + i_v * BLOCK_DV
        if ATOMIC:
            da_p = DA + bid * stride_b
            dk_p = DGK + bid * stride_b
            de_p = DE + bid * stride_b
            dq_p = DQ + bid * stride_b
        else:
            da_p = DA + pid * L * DK
            dk_p = DGK + pid * L * DK
            de_p = DE + pid * L * DK
            dq_p = DQ + pid * L * DK
        dz_p = DZ + bid * stride_bv + i_v * BLOCK_DV
        nblk = (L + BLOCK_T - 1) // BLOCK_T
        sb_p = SB + (bid * nblk) * DV * DK + i_v * BLOCK_DV * DK
        m2d = mv[:, None] & mk[None, :]

        dH = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)
        idx = tl.arange(0, BLOCK_T)
        for bii in range(nblk):
            bi = nblk - 1 - bii
            start = bi * BLOCK_T
            H_bnd = tl.load(
                sb_p + bi * DV * DK + rv[:, None] * DK + rk[None, :],
                mask=m2d, other=0.0,
            )
            H = H_bnd
            Hs = tl.zeros((BLOCK_T, BLOCK_DV, BLOCK_DK), dtype=tl.float32)
            for ti in tl.static_range(BLOCK_T):
                tt = start + ti
                lm = tt < L
                a_t = tl.load(a_p + tt * DK + rk, mask=mk & lm, other=1.0)
                k_t = tl.load(k_p + tt * DK + rk, mask=mk & lm, other=0.0)
                e_t = tl.load(e_p + tt * DK + rk, mask=mk & lm, other=0.0)
                z_t = tl.load(z_p + tt * DV + rv, mask=mv & lm, other=0.0)
                H = a_t[None, :] * H
                diff = z_t - tl.sum(H * e_t[None, :], axis=1)
                H = H + diff[:, None] * k_t[None, :]
                Hs = tl.where(idx[:, None, None] == ti, H[None, :, :], Hs)
            for ti in tl.static_range(BLOCK_T - 1, -1, -1):
                tt = start + ti
                lm = tt < L
                Ht = tl.sum(tl.where(idx[:, None, None] == ti, Hs, 0.0), axis=0)
                if ti > 0:
                    H_prev = tl.sum(
                        tl.where(idx[:, None, None] == (ti - 1), Hs, 0.0),
                        axis=0,
                    )
                else:
                    H_prev = H_bnd
                a_t = tl.load(a_p + tt * DK + rk, mask=mk & lm, other=1.0)
                k_t = tl.load(k_p + tt * DK + rk, mask=mk & lm, other=0.0)
                e_t = tl.load(e_p + tt * DK + rk, mask=mk & lm, other=0.0)
                q_t = tl.load(q_p + tt * DK + rk, mask=mk & lm, other=0.0)
                z_t = tl.load(z_p + tt * DV + rv, mask=mv & lm, other=0.0)
                go_t = tl.load(go_p + tt * DV + rv, mask=mv & lm, other=0.0)
                dH = dH + q_t[:, None] * go_t[None, :]
                diff = z_t - tl.sum(H_prev * (a_t * e_t)[None, :], axis=1)
                dk_t = tl.sum(dH * diff[None, :], axis=1)
                dd = tl.sum(dH * k_t[:, None], axis=0)
                dH = dH - e_t[:, None] * dd[None, :]
                de_t = -a_t * tl.sum(H_prev * dd[:, None], axis=0)
                dq_t = tl.sum(Ht * go_t[:, None], axis=0)
                da_t = tl.sum(dH * tl.trans(H_prev), axis=1)
                dH = a_t[:, None] * dH
                m1 = mk & lm
                if ATOMIC:
                    tl.atomic_add(
                        da_p + tt * DK + rk, da_t, mask=m1, sem="relaxed"
                    )
                    tl.atomic_add(
                        dk_p + tt * DK + rk, dk_t, mask=m1, sem="relaxed"
                    )
                    tl.atomic_add(
                        de_p + tt * DK + rk, de_t, mask=m1, sem="relaxed"
                    )
                    tl.atomic_add(
                        dq_p + tt * DK + rk, dq_t, mask=m1, sem="relaxed"
                    )
                else:
                    tl.store(da_p + tt * DK + rk, da_t, mask=m1)
                    tl.store(dk_p + tt * DK + rk, dk_t, mask=m1)
                    tl.store(de_p + tt * DK + rk, de_t, mask=m1)
                    tl.store(dq_p + tt * DK + rk, dq_t, mask=m1)
                tl.store(dz_p + tt * DV + rv, dd, mask=mv & lm)


def _can_use_recurrent(a, k, q, z):
    if not (_HAS_TRITON and a.is_cuda):
        return False
    if a.dtype != torch.float32:
        return False
    dk = k.shape[-2]
    dv = z.shape[-1]
    return dk <= _MAX_DIM and dv <= _MAX_DIM


def _blocks(DK, DV):
    return _TUNE.get((DK, DV), _TUNE_DEFAULT)


class FastRecurrentFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, k, e, q, z):
        ctx.shapes = [a.shape, k.shape, e.shape, q.shape, z.shape]
        with torch.amp.autocast(a.device.type, enabled=False):
            a = a.to(torch.float32).squeeze(-1).contiguous()
            k = k.to(torch.float32).squeeze(-1).contiguous()
            e = e.to(torch.float32).squeeze(-1).contiguous()
            q = q.to(torch.float32).squeeze(-1).contiguous()
            z = z.to(torch.float32).contiguous()
            B, L, DK = k.shape
            DV = z.shape[-1]
            O = torch.empty((B, L, DV), device=a.device, dtype=torch.float32)
            BDK = triton.next_power_of_2(DK)
            BV, nwarps = _blocks(DK, DV)
            BDFV = min(triton.next_power_of_2(DV), BV)
            NV = triton.cdiv(DV, BDFV)
            nblk = (L + _BLOCK_T - 1) // _BLOCK_T
            SB = torch.empty(
                (B, nblk, DV, DK), device=a.device, dtype=torch.float32
            )
            _gdr_fwd[(B * NV,)](
                a, k, e, q, z, O, SB, L, L * DK, L * DV,
                DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDFV,
                BLOCK_T=_BLOCK_T, num_warps=nwarps,
            )
            ctx.nv = NV
            ctx.bdfv = BDFV
            ctx.nwarps = nwarps
            ctx.save_for_backward(a, k, e, q, z, SB)
            return O

    @staticmethod
    def backward(ctx, go):
        with torch.amp.autocast(go.device.type, enabled=False):
            a, k, e, q, z, SB = ctx.saved_tensors
            B, L, DK = k.shape
            DV = z.shape[-1]
            go = go.contiguous()
            BDK = triton.next_power_of_2(DK)
            NV = ctx.nv
            # Direct atomic accumulation into (B, L, DK) avoids the per-tile
            # buffers and the four reduction kernels, but at short sequences
            # the atomics are slower than a plain store + reduction.  Use the
            # buffer path while its footprint stays small.
            atomic = 4 * B * NV * L * DK * 4 >= _ATOMIC_BYTES
            shape = (B, L, DK) if atomic else (B * NV, L, DK)
            da = torch.zeros(shape, device=a.device, dtype=torch.float32) if atomic \
                else torch.empty(shape, device=a.device, dtype=torch.float32)
            dk = torch.zeros(shape, device=a.device, dtype=torch.float32) if atomic \
                else torch.empty(shape, device=a.device, dtype=torch.float32)
            de = torch.zeros(shape, device=a.device, dtype=torch.float32) if atomic \
                else torch.empty(shape, device=a.device, dtype=torch.float32)
            dq = torch.zeros(shape, device=a.device, dtype=torch.float32) if atomic \
                else torch.empty(shape, device=a.device, dtype=torch.float32)
            dz = torch.empty_like(z)
            _gdr_bwd[(B * NV,)](
                a, k, e, q, z, go, da, dk, de, dq, dz, SB,
                L, L * DK, L * DV,
                DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=ctx.bdfv,
                BLOCK_T=_BLOCK_T, num_warps=ctx.nwarps, ATOMIC=atomic,
            )
            if not atomic:
                da = da.reshape(B, NV, L, DK).sum(1)
                dk = dk.reshape(B, NV, L, DK).sum(1)
                de = de.reshape(B, NV, L, DK).sum(1)
                dq = dq.reshape(B, NV, L, DK).sum(1)
            sa, sk, se, sq, sz = ctx.shapes
            return (
                da.reshape(sa),
                dk.reshape(sk),
                de.reshape(se),
                dq.reshape(sq),
                dz.reshape(sz),
            )
