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


def _can_split(B, L, DK, DV):
    """Best backend for (rows B, seqlen L, K-dim DK, V-dim DV) on CUDA.

    Data-driven dispatch: optuna-tuned on a 432-config grid
    (qk/vd/dim in {16..64}, B x heads in {1..64}, L in {128..4096},
    fp32/bf16, eager) with the objective mean(t_chosen / t_gd21) so every
    grid point weighs equally.

    Rows B = batch * heads (heads are moved to batch by the module).
    Split wins for long sequences; plain fused kernels win for short
    sequences and for wide K/V state tiles where the split's per-segment
    (DK x DK) map spills registers / the extra passes do not pay off:

      * L < 1024            -> plain (split overhead dominates at L <= 512;
                               ~neutral at L = 512-1023 on few rows).
      * DK or DV < 16       -> plain (split tiling requires >= 16).
      * DK >= 64 and DV >= 64 -> plain (measured x0.77-0.92 everywhere).
      * DV >= 64 and B >= 32 -> plain (many rows + wide V: x0.87).
      * DK >= 64 and B >= 32 -> plain (many rows + wide K: x0.89-0.92).
      * otherwise (L >= 1024, narrow-enough tiles, any row count) -> split:
          up to x1.3-4 faster than plain at L >= 1024 for few rows,
          x1.05-1.36 at 8 rows, and still x1.04-1.06 at 64 rows on
          DK=DV<=32.
    """
    R = B  # rows = batch * heads already folded into the batch axis
    if L < 1024 or DK < 16 or DV < 16:
        return False
    if DK >= 64 and DV >= 64:
        return False
    if DV >= 64 and R >= 32:
        return False
    if DK >= 64 and R >= 32:
        return False
    return True


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
            if _can_split(B, L, DK, DV):
                G = (L + _SEG - 1) // _SEG
                BDK = triton.next_power_of_2(DK)
                BDFV = min(triton.next_power_of_2(DV), _FWD_BV)
                NV = triton.cdiv(DV, BDFV)
                MSEG = torch.empty(
                    (B, G, DK, DK), device=a.device, dtype=torch.float32
                )
                CSEG = torch.empty(
                    (B, G, DK, DV), device=a.device, dtype=torch.float32
                )
                QT = torch.empty((B, L, DK), device=a.device, dtype=torch.float32)
                BUF = torch.empty((B, L, DV), device=a.device, dtype=torch.float32)
                SIN = torch.empty(
                    (B, G, DK, DV), device=a.device, dtype=torch.float32
                )
                _gdr_seg_fwd[(B * G * NV,)](
                    a, k, e, q, z, MSEG, CSEG, QT, BUF, L, S=_SEG,
                    stride_b=L * DK, stride_bv=L * DV,
                    DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDFV,
                    num_warps=1,
                )
                BDS = triton.next_power_of_2(DV)
                _gdr_scan_fwd[(B,)](
                    MSEG, CSEG, QT, BUF, O, SIN, L, S=_SEG,
                    stride_b=L * DK, stride_bv=L * DV,
                    DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDS,
                    num_warps=2,
                )
                ctx.split = True
                ctx.save_for_backward(a, k, e, q, z, MSEG, SIN)
                return O
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
            ctx.split = False
            ctx.save_for_backward(a, k, e, q, z, SB)
            return O

    @staticmethod
    def backward(ctx, go):
        with torch.amp.autocast(go.device.type, enabled=False):
            if ctx.split:
                a, k, e, q, z, MSEG, SIN = ctx.saved_tensors
                return RecurrentDelta2Fn._backward_split(
                    ctx, go, a, k, e, q, z, MSEG, SIN
                )
            a, k, e, q, z, SB = ctx.saved_tensors
            return RecurrentDelta2Fn._backward_plain(ctx, go, a, k, e, q, z, SB)

    @staticmethod
    def _backward_split(ctx, go, a, k, e, q, z, MSEG, SIN):
        B, L, DK = k.shape
        DV = z.shape[-1]
        G = (L + _SEG - 1) // _SEG
        go = go.contiguous()
        BDK = triton.next_power_of_2(DK)
        BDBV = min(triton.next_power_of_2(DV), _BWD_BV)
        NV = triton.cdiv(DV, BDBV)
        BG = torch.empty((B, G, DK, DV), device=a.device, dtype=torch.float32)
        DSIN = torch.empty((B, G, DK, DV), device=a.device, dtype=torch.float32)
        nblk_s = _SEG // _BLOCK_T
        SBSEG = torch.empty(
            (B * NV * G, nblk_s, BDBV, BDK),
            device=a.device, dtype=torch.float32,
        )
        da = torch.empty(
            (B * NV * G, _SEG, DK), device=a.device, dtype=torch.float32
        )
        dk = torch.empty(
            (B * NV * G, _SEG, DK), device=a.device, dtype=torch.float32
        )
        de = torch.empty(
            (B * NV * G, _SEG, DK), device=a.device, dtype=torch.float32
        )
        dq = torch.empty(
            (B * NV * G, _SEG, DK), device=a.device, dtype=torch.float32
        )
        dz = torch.empty_like(z)
        _gdr_seg_bwd_loc[(B * G * NV,)](
            a, k, e, q, go, BG, L, S=_SEG,
            stride_b=L * DK, stride_bv=L * DV,
            DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDBV,
            num_warps=1,
        )
        BDS = triton.next_power_of_2(DV)
        _gdr_scan_bwd[(B,)](
            MSEG, BG, DSIN, L, S=_SEG,
            DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDS,
            num_warps=2,
        )
        _gdr_seg_bwd_grads[(B * G * NV,)](
            a, k, e, q, z, go, SIN, DSIN, da, dk, de, dq, dz, SBSEG,
            L, S=_SEG,
            stride_b=L * DK, stride_bv=L * DV,
            DK=DK, DV=DV, BLOCK_DK=BDK, BLOCK_DV=BDBV, BLOCK_T=_BLOCK_T,
            num_warps=1,
        )
        da = da.reshape(B, G, NV, _SEG, DK).sum(2).reshape(B, G * _SEG, DK)[:, :L]
        dk = dk.reshape(B, G, NV, _SEG, DK).sum(2).reshape(B, G * _SEG, DK)[:, :L]
        de = de.reshape(B, G, NV, _SEG, DK).sum(2).reshape(B, G * _SEG, DK)[:, :L]
        dq = dq.reshape(B, G, NV, _SEG, DK).sum(2).reshape(B, G * _SEG, DK)[:, :L]
        sa, sk, se, sq, sz = ctx.shapes
        return (
            da.reshape(sa),
            dk.reshape(sk),
            de.reshape(se),
            dq.reshape(sq),
            dz.reshape(sz),
        )

    @staticmethod
    def _backward_plain(ctx, go, a, k, e, q, z, SB):
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

# ---------------------------------------------------------------------------
# Two-level (split-recurrent) path: the sequence is cut into _SEG-length
# segments; a first kernel computes each segment's state-transition map and
# per-token output coefficients (parallel over segments), a second kernel
# runs the tiny segment scan + outputs, and the backward mirrors this with a
# local-adjoint kernel, a reversed segment scan and a gradient kernel.
# ---------------------------------------------------------------------------

_SEG = 128
_TOK_BLK = tl.constexpr(16)


@triton.jit
def _gdr_seg_fwd(
    A, K, E, Q, Z, MSEG, CSEG, QT, BUF, L, S: tl.constexpr,
    stride_b, stride_bv,
    DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(DV, BLOCK_DV)
    G = (L + S - 1) // S
    i_v = pid % NV
    seg = (pid // NV) % G
    bid = pid // (NV * G)
    s0 = seg * S
    s1 = tl.minimum(s0 + S, L)
    rk = tl.arange(0, BLOCK_DK)
    rv = tl.arange(0, BLOCK_DV)
    mk = rk < DK
    mv = i_v * BLOCK_DV + rv < DV
    a_p = A + bid * stride_b + s0 * DK
    k_p = K + bid * stride_b + s0 * DK
    e_p = E + bid * stride_b + s0 * DK
    q_p = Q + bid * stride_b + s0 * DK
    z_p = Z + bid * stride_bv + i_v * BLOCK_DV + s0 * DV
    qt_p = QT + bid * L * DK + s0 * DK
    bf_p = BUF + bid * L * DV + i_v * BLOCK_DV + s0 * DV
    M = tl.where(
        (rk[:, None] == rk[None, :]) & mk[:, None] & mk[None, :], 1.0, 0.0
    )
    c = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)
    for t in range(S):
        tt = s0 + t
        lm = tt < s1
        a_t = tl.load(a_p + t * DK + rk, mask=mk & lm, other=1.0)
        k_t = tl.load(k_p + t * DK + rk, mask=mk & lm, other=0.0)
        e_t = tl.load(e_p + t * DK + rk, mask=mk & lm, other=0.0)
        q_t = tl.load(q_p + t * DK + rk, mask=mk & lm, other=0.0)
        z_t = tl.load(z_p + t * DV + rv, mask=mv & lm, other=0.0)
        M = a_t[:, None] * M
        rM = tl.sum(M * e_t[:, None], axis=0)
        M = M - k_t[:, None] * rM[None, :]
        c = a_t[:, None] * c
        rc = tl.sum(c * e_t[:, None], axis=0)
        c = c + k_t[:, None] * (z_t - rc)[None, :]
        qt_t = tl.sum(M * q_t[:, None], axis=0)
        b_t = tl.sum(c * q_t[:, None], axis=0)
        tl.store(qt_p + t * DK + rk, qt_t, mask=mk & lm)
        tl.store(bf_p + t * DV + rv, b_t, mask=mv & lm)
    tl.store(
        MSEG + (bid * G + seg) * DK * DK + rk[:, None] * DK + rk[None, :],
        M, mask=mk[:, None] & mk[None, :],
    )
    tl.store(
        CSEG + (bid * G + seg) * DK * DV + rk[:, None] * DV
        + (i_v * BLOCK_DV + rv)[None, :],
        c, mask=mk[:, None] & mv[None, :],
    )


@triton.jit
def _gdr_scan_fwd(
    MSEG, CSEG, QT, BUF, O, SIN, L, S: tl.constexpr,
    stride_b, stride_bv,
    DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    bid = tl.program_id(0)
    G = (L + S - 1) // S
    rk = tl.arange(0, BLOCK_DK)
    rv = tl.arange(0, BLOCK_DV)
    tj = tl.arange(0, _TOK_BLK)
    mk = rk < DK
    mv = rv < DV
    q_p = QT + bid * L * DK
    b_p = BUF + bid * L * DV
    o_p = O + bid * stride_bv
    sin_p = SIN + bid * G * DK * DV
    S_cur = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)
    for g in range(G):
        s0 = g * S
        s1 = tl.minimum(s0 + S, L)
        tl.store(
            sin_p + g * DK * DV + rk[:, None] * DV + rv[None, :],
            S_cur, mask=mk[:, None] & mv[None, :],
        )
        for tb in range(S // _TOK_BLK):
            tt = s0 + tb * _TOK_BLK
            lm = tt + tj < s1
            qb_t = tl.load(
                q_p + (tt + tj)[:, None] * DK + rk[None, :],
                mask=lm[:, None] & mk[None, :], other=0.0,
            )
            bb_t = tl.load(
                b_p + (tt + tj)[:, None] * DV + rv[None, :],
                mask=lm[:, None] & mv[None, :], other=0.0,
            )
            ob = tl.dot(tl.trans(S_cur), tl.trans(qb_t), input_precision="ieee") + tl.trans(bb_t)
            tl.store(
                o_p + (tt + tj)[:, None] * DV + rv[None, :],
                tl.trans(ob), mask=lm[:, None] & mv[None, :],
            )
        if s1 > s0:
            M_g = tl.load(
                MSEG + (bid * G + g) * DK * DK + rk[:, None] * DK + rk[None, :],
                mask=mk[:, None] & mk[None, :], other=0.0,
            )
            c_g = tl.load(
                CSEG + (bid * G + g) * DK * DV + rk[:, None] * DV + rv[None, :],
                mask=mk[:, None] & mv[None, :], other=0.0,
            )
            S_cur = tl.dot(M_g, S_cur, input_precision="ieee") + c_g


@triton.jit
def _gdr_seg_bwd_loc(
    A, K, E, Q, GO, BG, L, S: tl.constexpr,
    stride_b, stride_bv,
    DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(DV, BLOCK_DV)
    G = (L + S - 1) // S
    i_v = pid % NV
    seg = (pid // NV) % G
    bid = pid // (NV * G)
    s0 = seg * S
    s1 = tl.minimum(s0 + S, L)
    rk = tl.arange(0, BLOCK_DK)
    rv = tl.arange(0, BLOCK_DV)
    mk = rk < DK
    mv = i_v * BLOCK_DV + rv < DV
    a_p = A + bid * stride_b + s0 * DK
    k_p = K + bid * stride_b + s0 * DK
    e_p = E + bid * stride_b + s0 * DK
    q_p = Q + bid * stride_b + s0 * DK
    go_p = GO + bid * stride_bv + i_v * BLOCK_DV + s0 * DV
    dS = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)
    for t in tl.static_range(S - 1, -1, -1):
        tt = s0 + t
        lm = tt < s1
        a_t = tl.load(a_p + t * DK + rk, mask=mk & lm, other=1.0)
        k_t = tl.load(k_p + t * DK + rk, mask=mk & lm, other=0.0)
        e_t = tl.load(e_p + t * DK + rk, mask=mk & lm, other=0.0)
        q_t = tl.load(q_p + t * DK + rk, mask=mk & lm, other=0.0)
        go_t = tl.load(go_p + t * DV + rv, mask=mv & lm, other=0.0)
        dS = dS + q_t[:, None] * go_t[None, :]
        dd = tl.sum(dS * k_t[:, None], axis=0)
        dS = dS - e_t[:, None] * dd[None, :]
        dS = a_t[:, None] * dS
    tl.store(
        BG + (bid * G + seg) * DK * DV + rk[:, None] * DV
        + (i_v * BLOCK_DV + rv)[None, :],
        dS, mask=mk[:, None] & mv[None, :],
    )


@triton.jit
def _gdr_scan_bwd(
    MSEG, BG, DSIN, L, S: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    bid = tl.program_id(0)
    G = (L + S - 1) // S
    rk = tl.arange(0, BLOCK_DK)
    rv = tl.arange(0, BLOCK_DV)
    mk = rk < DK
    mv = rv < DV
    m2d = mk[:, None] & mv[None, :]
    dS = tl.zeros((BLOCK_DK, BLOCK_DV), dtype=tl.float32)
    for gi in range(G):
        g = G - 1 - gi
        s1 = tl.minimum((g + 1) * S, L)
        if s1 > g * S:
            M_g = tl.load(
                MSEG + (bid * G + g) * DK * DK + rk[:, None] * DK + rk[None, :],
                mask=mk[:, None] & mk[None, :], other=0.0,
            )
            b_g = tl.load(
                BG + (bid * G + g) * DK * DV + rk[:, None] * DV + rv[None, :],
                mask=m2d, other=0.0,
            )
            dS = tl.dot(tl.trans(M_g), dS, input_precision="ieee") + b_g
        tl.store(
            DSIN + (bid * G + g) * DK * DV + rk[:, None] * DV + rv[None, :],
            dS, mask=m2d,
        )


@triton.jit
def _gdr_seg_bwd_grads(
    A, K, E, Q, Z, GO, SIN, DSIN, DA, DGK, DE, DQ, DZ, SBSEG,
    L, S: tl.constexpr,
    stride_b, stride_bv,
    DK: tl.constexpr, DV: tl.constexpr,
    BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(DV, BLOCK_DV)
    G = (L + S - 1) // S
    i_v = pid % NV
    seg = (pid // NV) % G
    bid = pid // (NV * G)
    s0 = seg * S
    s1 = tl.minimum(s0 + S, L)
    rk = tl.arange(0, BLOCK_DK)
    rv = tl.arange(0, BLOCK_DV)
    mk = rk < DK
    mv = i_v * BLOCK_DV + rv < DV
    m2d = mv[:, None] & mk[None, :]
    a_p = A + bid * stride_b + s0 * DK
    k_p = K + bid * stride_b + s0 * DK
    e_p = E + bid * stride_b + s0 * DK
    q_p = Q + bid * stride_b + s0 * DK
    z_p = Z + bid * stride_bv + i_v * BLOCK_DV + s0 * DV
    go_p = GO + bid * stride_bv + i_v * BLOCK_DV + s0 * DV
    da_p = DA + (pid * S) * DK
    dk_p = DGK + (pid * S) * DK
    de_p = DE + (pid * S) * DK
    dq_p = DQ + (pid * S) * DK
    dz_p = DZ + bid * stride_bv + i_v * BLOCK_DV + s0 * DV
    nblk_s = S // BLOCK_T
    sb_p = SBSEG + (pid * nblk_s) * BLOCK_DV * BLOCK_DK
    sin_p = SIN + (bid * G + seg) * DK * DV
    dsin_p = DSIN + (bid * G + tl.minimum(seg + 1, G - 1)) * DK * DV
    H_bnd = tl.load(
        sin_p + (i_v * BLOCK_DV + rv)[:, None] + rk[None, :] * DV,
        mask=m2d, other=0.0,
    )

    # boundary pass: local forward states at BLOCK_T multiples
    H = H_bnd
    for t in range(S):
        if t % BLOCK_T == 0:
            tl.store(
                sb_p + (t // BLOCK_T) * BLOCK_DV * BLOCK_DK
                + rv[:, None] * DK + rk[None, :],
                H, mask=m2d,
            )
        tt = s0 + t
        lm = tt < s1
        a_t = tl.load(a_p + t * DK + rk, mask=mk & lm, other=1.0)
        k_t = tl.load(k_p + t * DK + rk, mask=mk & lm, other=0.0)
        e_t = tl.load(e_p + t * DK + rk, mask=mk & lm, other=0.0)
        z_t = tl.load(z_p + t * DV + rv, mask=mv & lm, other=0.0)
        H = a_t[None, :] * H
        diff = z_t - tl.sum(H * e_t[None, :], axis=1)
        H = H + diff[:, None] * k_t[None, :]

    # adjoint, block by block (reversed within the segment)
    dH = tl.load(
        dsin_p + rk[:, None] * DV + (i_v * BLOCK_DV + rv)[None, :],
        mask=mk[:, None] & mv[None, :] & (seg < G - 1), other=0.0,
    )
    idx = tl.arange(0, BLOCK_T)
    for bii in range(nblk_s):
        bi = nblk_s - 1 - bii
        start = bi * BLOCK_T
        H_bnd2 = tl.load(
            sb_p + bi * BLOCK_DV * BLOCK_DK + rv[:, None] * DK + rk[None, :],
            mask=m2d, other=0.0,
        )
        H = H_bnd2
        Hs = tl.zeros((BLOCK_T, BLOCK_DV, BLOCK_DK), dtype=tl.float32)
        for ti in tl.static_range(BLOCK_T):
            tt = start + ti
            lm = tt < s1 - s0
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
            lm = tt < s1 - s0
            Ht = tl.sum(tl.where(idx[:, None, None] == ti, Hs, 0.0), axis=0)
            if ti > 0:
                H_prev = tl.sum(
                    tl.where(idx[:, None, None] == (ti - 1), Hs, 0.0), axis=0
                )
            else:
                H_prev = H_bnd2
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
