"""Fused per-token recurrence kernels for the Gated Delta Rule-2.

The chunked WY formulation parallelizes the recurrence over chunks, but at
small/medium sequence lengths the per-chunk matmuls + triangular solves are
launch-bound. These Triton kernels instead run the exact per-token
recurrence (identical math to the serial reference) with the state held in
registers: the state is kept transposed (V-rows x K-cols), tiled over the V
dimension with small blocks (FLA fused_recurrent style). No WY solve, no
chunk scan, no decay-range limitations (hard resets are exact).
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # CPU-only torch without triton
    _HAS_TRITON = False

_BLOCK_T = 8
_MAX_DIM = 256
_FWD_BV = 8
_BWD_BV = 8


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
                sb_p + (t // BLOCK_T) * DV * DK + rv[:, None] * DK + rk[None, :],
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
    go_p = GO + bid * stride_bv + i_v * BLOCK_DV
    da_p = DA + pid * L * DK
    dk_p = DGK + pid * L * DK
    de_p = DE + pid * L * DK
    dq_p = DQ + pid * L * DK
    dz_p = DZ + bid * stride_bv + i_v * BLOCK_DV
    nblk = (L + BLOCK_T - 1) // BLOCK_T
    sb_p = SB + (bid * nblk) * DV * DK + i_v * BLOCK_DV * DK
    m2d = mv[:, None] & mk[None, :]

    # adjoint recurrence, block by block (reversed); the forward states at
    # block boundaries were checkpointed by the forward kernel into SB
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
                    tl.where(idx[:, None, None] == (ti - 1), Hs, 0.0), axis=0
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
            tl.store(da_p + tt * DK + rk, da_t, mask=mk & lm)
            tl.store(dk_p + tt * DK + rk, dk_t, mask=mk & lm)
            tl.store(de_p + tt * DK + rk, de_t, mask=mk & lm)
            tl.store(dq_p + tt * DK + rk, dq_t, mask=mk & lm)
            tl.store(dz_p + tt * DV + rv, dd, mask=mv & lm)


def _can_use_recurrent(a, k, q, z):
    if not (_HAS_TRITON and a.is_cuda):
        return False
    if a.dtype != torch.float32:
        return False
    dk = k.shape[-2]
    dv = z.shape[-1]
    return dk <= _MAX_DIM and dv <= _MAX_DIM


class RecurrentDelta2Fn(torch.autograd.Function):
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
            BDFV = min(triton.next_power_of_2(DV), _FWD_BV)
            NV = triton.cdiv(DV, BDFV)
            nblk = (L + _BLOCK_T - 1) // _BLOCK_T
            SB = torch.empty(
                (B, nblk, DV, DK), device=a.device, dtype=torch.float32
            )
            _gdr_fwd[(B * NV,)](
                a, k, e, q, z, O, SB, L, L * DK, L * DV,
                DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDFV,
                BLOCK_T=_BLOCK_T,
                num_warps=1,
            )
            ctx.save_for_backward(a, k, e, q, z, SB)
            return O

    @staticmethod
    def backward(ctx, go):
        a, k, e, q, z, SB = ctx.saved_tensors
        with torch.amp.autocast(go.device.type, enabled=False):
            B, L, DK = k.shape
            DV = z.shape[-1]
            go = go.contiguous()
            BDK = triton.next_power_of_2(DK)
            BDBV = min(triton.next_power_of_2(DV), _BWD_BV)
            NV = triton.cdiv(DV, BDBV)
            da = torch.empty((B * NV, L, DK), device=a.device, dtype=torch.float32)
            dk = torch.empty((B * NV, L, DK), device=a.device, dtype=torch.float32)
            de = torch.empty((B * NV, L, DK), device=a.device, dtype=torch.float32)
            dq = torch.empty((B * NV, L, DK), device=a.device, dtype=torch.float32)
            dz = torch.empty_like(z)
            _gdr_bwd[(B * NV,)](
                a, k, e, q, z, go, da, dk, de, dq, dz, SB,
                L, L * DK, L * DV,
                DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDBV, BLOCK_T=_BLOCK_T,
                num_warps=1,
            )
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
