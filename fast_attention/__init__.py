"""Fused Triton SelfAttention/CrossAttention.

Replaces the eager SDPA region (permute copies + cutlass sm80 f32 mem-efficient
attention fwd/bwd + xsa + to_out conv + layout copies) with:
  - rotate kernel: RoPE on the [B,{3|2},H,D,L] conv output (no layout copies)
  - attn fwd:   flash-attention fwd + xsa + to_out GEMM + bias + residual,
                writes the final [B, dim, H, W] output directly
  - attn bwd:   dY@W^T (chunked) + flash-attention bwd (2-pass, LSE recompute)
                + xsa backward + dW_out/dbias accumulation
  - unrotate kernel: inverse RoPE on dq/dk gradients

All dot products run on tensor cores with tf32 (fp32 accumulate), within the
1e-2 tolerance gate vs eager fp32.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from kemsekov_torch.attention import SelfAttention, CrossAttention



# ---------------------------------------------------------------------------
# RoPE rotation kernel.
# x is [B, P, Hh, D, L] contiguous (P = 1, 2 or 3 parts); parts 0,1 rotated.
# rows enumerate (b, part in {0..min(P,2)-1}, head, l)
# ---------------------------------------------------------------------------

@triton.jit
def _rot_kernel(
    x_ptr, o_ptr,
    B, Hh, D, L, sp1, sp2,
    cos0, sin0, cos1, sin1, cos2, sin2,
    q, R,
    P: tl.constexpr, NP: tl.constexpr, A: tl.constexpr, INVERSE: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_I: tl.constexpr,
):
    """INVERSE=False: x [B,P,H,D,L] -> o [B,P,H,L,D] (fwd)
    INVERSE=True:  x [B,P,H,L,D] -> o [B,P,H,D,L] (bwd)"""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    m_r = rows < R
    m_i = i < A * q
    m = m_r[:, None] & m_i[None, :]

    l = rows % L
    pos1 = (l // sp2) % sp1
    pos0 = l // (sp1 * sp2)
    pos2 = l % sp2
    part = (rows // L) % NP
    h = (rows // (NP * L)) % Hh
    b = rows // (NP * L * Hh)

    i2 = i % q
    a = i // q
    c1 = 2 * a * q + i2
    c2 = c1 + q
    if INVERSE:
        in_row = b * (P * Hh * L * D) + part * (Hh * L * D) + h * (L * D) + l * D
        out_row = b * (P * Hh * D * L) + part * (Hh * D * L) + h * (D * L) + l
        offs1 = in_row[:, None] + c1[None, :]
        offs2 = in_row[:, None] + c2[None, :]
        o1 = out_row[:, None] + c1[None, :] * L
        o2 = out_row[:, None] + c2[None, :] * L
    else:
        in_row = b * (P * Hh * D * L) + part * (Hh * D * L) + h * (D * L) + l
        out_row = b * (P * Hh * L * D) + part * (Hh * L * D) + h * (L * D) + l * D
        offs1 = in_row[:, None] + c1[None, :] * L
        offs2 = in_row[:, None] + c2[None, :] * L
        o1 = out_row[:, None] + c1[None, :]
        o2 = out_row[:, None] + c2[None, :]

    x1 = tl.load(x_ptr + offs1, mask=m, other=0.0)
    x2 = tl.load(x_ptr + offs2, mask=m, other=0.0)
    ct = tl.zeros([BLOCK_R, BLOCK_I], dtype=tl.float32)
    st = tl.zeros([BLOCK_R, BLOCK_I], dtype=tl.float32)
    for ax in tl.static_range(3):
        if ax == 0:
            c_ptr, s_ptr, pos = cos0, sin0, pos0
        elif ax == 1:
            c_ptr, s_ptr, pos = cos1, sin1, pos1
        else:
            c_ptr, s_ptr, pos = cos2, sin2, pos2
        sel = (a == ax)[None, :]
        idx = tl.where(sel, pos[:, None] * q + i2[None, :], 0)
        msel = m & sel
        ct = tl.where(sel, tl.load(c_ptr + idx, mask=msel, other=0.0), ct)
        st = tl.where(sel, tl.load(s_ptr + idx, mask=msel, other=0.0), st)

    if INVERSE:
        y1 = x1 * ct + x2 * st
        y2 = x2 * ct - x1 * st
    else:
        y1 = x1 * ct - x2 * st
        y2 = x1 * st + x2 * ct
    tl.store(o_ptr + o1, y1, mask=m)
    tl.store(o_ptr + o2, y2, mask=m)


# ---------------------------------------------------------------------------
# Fused attention forward: flash fwd + xsa; writes A (bwd) and A_x (outproj)
# ---------------------------------------------------------------------------

@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    A_ptr, LSE_ptr, Ax_ptr,
    s_qz, s_qh, s_qn, s_qd,
    s_kz, s_kh, s_kn, s_kd,
    s_vz, s_vh, s_vn, s_vd,
    B, H, D, Lq, Lk, inner,
    scale,
    XSA: tl.constexpr, CAUSAL: tl.constexpr, EVEN_N: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    b = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < Lq

    q = tl.load(
        q_ptr + b * s_qz + h * s_qh + offs_m[:, None] * s_qn + offs_d[None, :] * s_qd,
        mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), other=0.0)
    q = q.to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    lo = 0
    hi = Lk
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, Lk)

    for start_n in range(lo, hi, BLOCK_N):
        offs_nn = start_n + offs_n
        k = tl.load(
            k_ptr + b * s_kz + h * s_kh + offs_nn[:, None] * s_kn + offs_d[None, :] * s_kd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        qk = qk.to(tl.float32)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_nn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(offs_nn[None, :] < Lk, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_nn[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=PREC)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    A = acc / l_i[:, None]
    tl.store(A_ptr + (b * H + h) * Lq * D + offs_m[:, None] * D + offs_d[None, :],
             A, mask=mask_m[:, None])
    tl.store(LSE_ptr + (b * H + h) * Lq + offs_m, m_i + tl.log(l_i), mask=mask_m)

    if XSA:
        v_m = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_m[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        nrm = tl.sqrt(tl.sum(v_m * v_m, 1))
        vn = v_m / tl.maximum(nrm, 1e-12)[:, None]
        s = tl.sum(A * vn, 1)
        A_x = A - s[:, None] * vn
    else:
        A_x = A

    tl.store(Ax_ptr + b * Lq * inner + offs_m[:, None] * inner + h * D + offs_d[None, :],
             A_x, mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Out projection: out = A_x @ W^T + bias + identity  (full inner sum)
# ---------------------------------------------------------------------------

@triton.jit
def _outproj_kernel(
    A_ptr, W_ptr, b_ptr, iden_ptr, out_ptr,
    B, L, inner, dim,
    HAS_BIAS: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < L

    a_base = pid_b * (L * inner) + offs_m[:, None] * inner
    o_base = pid_b * (dim * L) + offs_m[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, inner, BLOCK_K):
        mask_k = k + offs_k < inner
        a = tl.load(A_ptr + a_base + (k + offs_k)[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W_ptr + (k + offs_k)[:, None] + offs_n[None, :] * inner,
                    mask=mask_k[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(a, w, input_precision=PREC)

    acc_t = tl.trans(acc)
    if HAS_BIAS:
        bv = tl.load(b_ptr + offs_n, mask=offs_n < dim, other=0.0)
        acc_t = acc_t + bv[:, None]
    iden_t = tl.load(iden_ptr + o_base + offs_n[:, None] * L,
                     mask=(offs_n[:, None] < dim) & mask_m[None, :], other=0.0)
    acc_t = acc_t + iden_t
    tl.store(out_ptr + o_base + offs_n[:, None] * L,
             acc_t, mask=(offs_n[:, None] < dim) & mask_m[None, :])


# ---------------------------------------------------------------------------
# Fused attention backward: dY@W^T (chunked) + xsa bwd + flash bwd + dW/dbias
# ---------------------------------------------------------------------------

@triton.jit
def _attn_bwd_kernel(
    q_ptr, k_ptr, v_ptr,
    dA_ptr, LSE_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    s_qz, s_qh, s_qn, s_qd,
    s_kz, s_kh, s_kn, s_kd,
    s_vz, s_vh, s_vn, s_vd,
    s_dqz, s_dqh, s_dqn, s_dqd,
    s_dkz, s_dkh, s_dkn, s_dkd,
    s_dvz, s_dvh, s_dvn, s_dvd,
    B, H, D, Lq, Lk,
    scale,
    XSA: tl.constexpr, CAUSAL: tl.constexpr, EVEN_N: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    b = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < Lq

    q = tl.load(
        q_ptr + b * s_qz + h * s_qh + offs_m[:, None] * s_qn + offs_d[None, :] * s_qd,
        mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    dA = tl.load(
        dA_ptr + (b * H + h) * Lq * D + offs_m[:, None] * D + offs_d[None, :],
        mask=mask_m[:, None], other=0.0)
    lse = tl.load(LSE_ptr + (b * H + h) * Lq + offs_m, mask=mask_m, other=0.0)

    # --- pass 1: D[m] = sum_n P(m,n) * (dA[m] . v[n]) ---------------------
    D_m = tl.zeros([BLOCK_M], dtype=tl.float32)
    lo = 0
    hi = Lk
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, Lk)
    for start_n in range(lo, hi, BLOCK_N):
        offs_nn = start_n + offs_n
        k = tl.load(
            k_ptr + b * s_kz + h * s_kh + offs_nn[:, None] * s_kn + offs_d[None, :] * s_kd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        qk = qk.to(tl.float32)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_nn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(offs_nn[None, :] < Lk, qk, float("-inf"))
        p = tl.exp(qk - lse[:, None])
        v = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_nn[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        dots = tl.dot(dA, tl.trans(v), input_precision=PREC)
        D_m += tl.sum(p * dots, 1)

    # --- pass 2: dq/dk/dv -------------------------------------------------
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        offs_nn = start_n + offs_n
        k = tl.load(
            k_ptr + b * s_kz + h * s_kh + offs_nn[:, None] * s_kn + offs_d[None, :] * s_kd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        qk = qk.to(tl.float32)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_nn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(offs_nn[None, :] < Lk, qk, float("-inf"))
        p = tl.exp(qk - lse[:, None])
        v = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_nn[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        dots = tl.dot(dA, tl.trans(v), input_precision=PREC)
        dS = (p * (dots - D_m[:, None]) * scale).to(tl.float32)
        dq += tl.dot(dS, k, input_precision=PREC)
        dk_part = tl.dot(tl.trans(dS), q, input_precision=PREC)
        dv_part = tl.dot(tl.trans(p), dA, input_precision=PREC)
        tl.atomic_add(dk_ptr + b * s_dkz + h * s_dkh + offs_nn[:, None] * s_dkn + offs_d[None, :] * s_dkd,
                      dk_part, mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), sem="relaxed")
        tl.atomic_add(dv_ptr + b * s_dvz + h * s_dvh + offs_nn[:, None] * s_dvn + offs_d[None, :] * s_dvd,
                      dv_part, mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), sem="relaxed")

    # dq is complete for this block — direct store
    tl.store(dq_ptr + b * s_dqz + h * s_dqh + offs_m[:, None] * s_dqn + offs_d[None, :] * s_dqd,
             dq, mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D))



# ---------------------------------------------------------------------------
# Single-kernel backward (small sequences: avoids extra launch overhead)
# ---------------------------------------------------------------------------

@triton.jit
def _attn_bwd_kernel_single(
    q_ptr, k_ptr, v_ptr,
    A_ptr, LSE_ptr, dout_ptr,
    dWp_ptr, dbias_ptr, dq_ptr, dk_ptr, dv_ptr,
    s_qz, s_qh, s_qn, s_qd,
    s_kz, s_kh, s_kn, s_kd,
    s_vz, s_vh, s_vn, s_vd,
    s_dqz, s_dqh, s_dqn, s_dqd,
    s_dkz, s_dkh, s_dkn, s_dkd,
    s_dvz, s_dvh, s_dvn, s_dvd,
    W_ptr,
    B, H, D, Lq, Lk, dim,
    scale,
    XSA: tl.constexpr, CAUSAL: tl.constexpr, EVEN_N: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    b = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, CHUNK_C)
    mask_m = offs_m < Lq

    q = tl.load(
        q_ptr + b * s_qz + h * s_qh + offs_m[:, None] * s_qn + offs_d[None, :] * s_qd,
        mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    dA = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for kc in range(0, dim, CHUNK_C):
        mask_c = kc + offs_c < dim
        dYc = tl.trans(tl.load(
            dout_ptr + b * dim * Lq + (kc + offs_c)[:, None] * Lq + offs_m[None, :],
            mask=mask_c[:, None] & mask_m[None, :], other=0.0)).to(tl.float32).to(tl.float32)
        Wc = tl.load(W_ptr + (kc + offs_c)[:, None] * dim + (h * D + offs_d)[None, :],
                     mask=mask_c[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        dA += tl.dot(dYc, Wc, input_precision=PREC)
        db = tl.sum(dYc, 0)
        tl.atomic_add(dbias_ptr + kc + offs_c, db, mask=mask_c, sem="relaxed")

    lse = tl.load(LSE_ptr + (b * H + h) * Lq + offs_m, mask=mask_m, other=0.0)
    A = tl.load(A_ptr + (b * H + h) * Lq * D + offs_m[:, None] * D + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)
    if XSA:
        v_m = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_m[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        nrm = tl.sqrt(tl.sum(v_m * v_m, 1))
        vn = v_m / tl.maximum(nrm, 1e-12)[:, None]
        s = tl.sum(A * vn, 1)
        dA_x = dA - tl.sum(dA * vn, 1)[:, None] * vn
        g = -tl.sum(dA * vn, 1)[:, None] * A - s[:, None] * dA
        Pg = g - vn * tl.sum(g * vn, 1)[:, None]
        dv_self = Pg / tl.maximum(nrm, 1e-12)[:, None]
        A_x = A - s[:, None] * vn
        dA = dA_x
    else:
        A_x = A
        dv_self = None

    D_m = tl.zeros([BLOCK_M], dtype=tl.float32)
    lo = 0
    hi = Lk
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, Lk)
    for start_n in range(lo, hi, BLOCK_N):
        offs_nn = start_n + offs_n
        k = tl.load(
            k_ptr + b * s_kz + h * s_kh + offs_nn[:, None] * s_kn + offs_d[None, :] * s_kd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        qk = qk.to(tl.float32)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_nn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(offs_nn[None, :] < Lk, qk, float("-inf"))
        p = tl.exp(qk - lse[:, None])
        v = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_nn[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        dots = tl.dot(dA, tl.trans(v), input_precision=PREC)
        D_m += tl.sum(p * dots, 1)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        offs_nn = start_n + offs_n
        k = tl.load(
            k_ptr + b * s_kz + h * s_kh + offs_nn[:, None] * s_kn + offs_d[None, :] * s_kd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
        qk = qk.to(tl.float32)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_nn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(offs_nn[None, :] < Lk, qk, float("-inf"))
        p = tl.exp(qk - lse[:, None])
        v = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_nn[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        dots = tl.dot(dA, tl.trans(v), input_precision=PREC)
        dS = (p * (dots - D_m[:, None]) * scale).to(tl.float32)
        dq += tl.dot(dS, k, input_precision=PREC)
        dk_part = tl.dot(tl.trans(dS), q, input_precision=PREC)
        dv_part = tl.dot(tl.trans(p), dA, input_precision=PREC)
        tl.atomic_add(dk_ptr + b * s_dkz + h * s_dkh + offs_nn[:, None] * s_dkn + offs_d[None, :] * s_dkd,
                      dk_part, mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), sem="relaxed")
        tl.atomic_add(dv_ptr + b * s_dvz + h * s_dvh + offs_nn[:, None] * s_dvn + offs_d[None, :] * s_dvd,
                      dv_part, mask=(offs_nn[:, None] < Lk) & (offs_d[None, :] < D), sem="relaxed")

    tl.store(dq_ptr + b * s_dqz + h * s_dqh + offs_m[:, None] * s_dqn + offs_d[None, :] * s_dqd,
             dq, mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D))

    if XSA:
        tl.atomic_add(dv_ptr + b * s_dvz + h * s_dvh + offs_m[:, None] * s_dvn + offs_d[None, :] * s_dvd,
                      dv_self, mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), sem="relaxed")

    for kc in range(0, dim, CHUNK_C):
        mask_c = kc + offs_c < dim
        dYc = tl.trans(tl.load(
            dout_ptr + b * dim * Lq + (kc + offs_c)[:, None] * Lq + offs_m[None, :],
            mask=mask_c[:, None] & mask_m[None, :], other=0.0)).to(tl.float32)
        dWc = tl.dot(tl.trans(A_x), dYc, input_precision=PREC)
        tl.atomic_add(dWp_ptr + (b * dim + kc + offs_c[None, :]) * H * D + h * D + offs_d[:, None],
                      dWc, mask=(offs_d[:, None] < D) & mask_c[None, :], sem="relaxed")


# ---------------------------------------------------------------------------
# Backward prep kernel: dA = dY @ W_h^T (chunked) + xsa projection +
#                        dW_out/dbias/dv_self accumulation
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_prep_kernel(
    A_ptr, LSE_ptr, dout_ptr, v_ptr,
    dWp_ptr, dbias_ptr, dAout_ptr, dv_ptr,
    s_vz, s_vh, s_vn, s_vd,
    s_dvz, s_dvh, s_dvn, s_dvd,
    W_ptr,
    B, H, D, Lq, Lk, dim,
    XSA: tl.constexpr,
    PREC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    b = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, CHUNK_C)
    mask_m = offs_m < Lq

    dA = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for kc in range(0, dim, CHUNK_C):
        mask_c = kc + offs_c < dim
        dYc = tl.trans(tl.load(
            dout_ptr + b * dim * Lq + (kc + offs_c)[:, None] * Lq + offs_m[None, :],
            mask=mask_c[:, None] & mask_m[None, :], other=0.0)).to(tl.float32).to(tl.float32)  # [BM, CC]
        Wc = tl.load(W_ptr + (kc + offs_c)[:, None] * dim + (h * D + offs_d)[None, :],
                     mask=mask_c[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)  # [CC, D]
        dA += tl.dot(dYc, Wc, input_precision=PREC)
        db = tl.sum(dYc, 0)
        tl.atomic_add(dbias_ptr + kc + offs_c, db, mask=mask_c, sem="relaxed")

    A = tl.load(A_ptr + (b * H + h) * Lq * D + offs_m[:, None] * D + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)
    if XSA:
        v_m = tl.load(
            v_ptr + b * s_vz + h * s_vh + offs_m[:, None] * s_vn + offs_d[None, :] * s_vd,
            mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
        nrm = tl.sqrt(tl.sum(v_m * v_m, 1))
        vn = v_m / tl.maximum(nrm, 1e-12)[:, None]
        s = tl.sum(A * vn, 1)
        dA_x = dA - tl.sum(dA * vn, 1)[:, None] * vn           # projection
        g = -tl.sum(dA * vn, 1)[:, None] * A - s[:, None] * dA  # dvhat gradient
        Pg = g - vn * tl.sum(g * vn, 1)[:, None]
        dv_self = Pg / tl.maximum(nrm, 1e-12)[:, None]
        A_x = A - s[:, None] * vn
        tl.atomic_add(dv_ptr + b * s_dvz + h * s_dvh + offs_m[:, None] * s_dvn + offs_d[None, :] * s_dvd,
                      dv_self, mask=(offs_m[:, None] < Lq) & (offs_d[None, :] < D), sem="relaxed")
        dA = dA_x
    else:
        A_x = A

    tl.store(dAout_ptr + (b * H + h) * Lq * D + offs_m[:, None] * D + offs_d[None, :],
             dA, mask=mask_m[:, None])

    for kc in range(0, dim, CHUNK_C):
        mask_c = kc + offs_c < dim
        dYc = tl.trans(tl.load(
            dout_ptr + b * dim * Lq + (kc + offs_c)[:, None] * Lq + offs_m[None, :],
            mask=mask_c[:, None] & mask_m[None, :], other=0.0)).to(tl.float32)
        dWc = tl.dot(tl.trans(A_x), dYc, input_precision=PREC)  # [D, CC]
        tl.atomic_add(dWp_ptr + (b * dim + kc + offs_c[None, :]) * H * D + h * D + offs_d[:, None],
                      dWc, mask=(offs_d[:, None] < D) & mask_c[None, :], sem="relaxed")


# ---------------------------------------------------------------------------
# The autograd function
# ---------------------------------------------------------------------------

_ZERO1 = None
_DUMMY = {}

def _zeros(dim, dev):
    global _ZERO1
    if _ZERO1 is None or _ZERO1.device != dev or _ZERO1.numel() != dim:
        _ZERO1 = torch.zeros(dim, device=dev)
    return _ZERO1


def _tab(t, dev):
    """None tables -> dummy (1,1) tensor (masked axes never load)."""
    if t is not None:
        return t
    key = dev
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(1, 1, device=dev)
    return _DUMMY[key]


class _FusedAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_in, kv_in, iden, w_out, b_out, plan, is_cross, xsa, is_causal, scale):
        B, H, D = plan["B"], plan["H"], plan["D"]
        Lq, Lk = plan["Lq"], plan["Lk"]
        dim = H * D
        dev = q_in.device
        q_in = q_in.contiguous()
        if kv_in is not None:
            kv_in = kv_in.contiguous()
        A = torch.empty(B, H, Lq, D, device=dev, dtype=torch.float32)
        lse = torch.empty(B, H, Lq, device=dev, dtype=torch.float32)
        out = torch.empty_like(iden)
        bias = b_out if b_out is not None else _zeros(dim, dev)

        kv_r = None
        if plan["rotate"]:
            cfg = _ROT_CFG
            q_ = plan["q"]
            qkv_r = torch.empty(B, 3 if not is_cross else 1, H, Lq, D, device=dev, dtype=q_in.dtype)
            R = B * (2 if not is_cross else 1) * H * Lq
            grid = (triton.cdiv(R, cfg["BR"]), triton.cdiv(plan["A"] * q_, cfg["BI"]))
            _rot_kernel[grid](
                q_in, qkv_r, B, H, D, Lq, plan["sp1"], plan["sp2"],
                _tab(plan["cos0"], dev), _tab(plan["sin0"], dev),
                _tab(plan["cos1"], dev), _tab(plan["sin1"], dev),
                _tab(plan["cos2"], dev), _tab(plan["sin2"], dev),
                q_, R, P=3 if not is_cross else 1, NP=2 if not is_cross else 1, A=plan["A"], INVERSE=False,
                BLOCK_R=cfg["BR"], BLOCK_I=cfg["BI"], num_warps=cfg["warp"],
            )
            if not is_cross:
                qkv_r[:, 2] = q_in.reshape(B, 3, H, D, Lq)[:, 2].transpose(-1, -2)
            if is_cross:
                kv_r = torch.empty(B, 2, H, Lk, D, device=dev, dtype=kv_in.dtype)
                R = B * H * Lk
                grid = (triton.cdiv(R, cfg["BR"]), triton.cdiv(plan["A"] * q_, cfg["BI"]))
                _rot_kernel[grid](
                    kv_in, kv_r, B, H, D, Lk, plan["sp1k"], plan["sp2k"],
                    _tab(plan["cos0k"], dev), _tab(plan["sin0k"], dev),
                    _tab(plan["cos1k"], dev), _tab(plan["sin1k"], dev),
                    _tab(plan["cos2k"], dev), _tab(plan["sin2k"], dev),
                    q_, R, P=2, NP=1, A=plan["A"], INVERSE=False,
                    BLOCK_R=cfg["BR"], BLOCK_I=cfg["BI"], num_warps=cfg["warp"],
                )
                kv_r[:, 1] = kv_in.reshape(B, 2, H, D, Lk)[:, 1].transpose(-1, -2)
        else:
            qkv_r = q_in.reshape(B, 3, H, D, Lq).transpose(-1, -2).contiguous()
            kv_r = kv_in

        if is_cross:
            q = qkv_r[:, 0]
            kv = kv_r.reshape(B, 2, H, Lk, D)
            k, v = kv[:, 0], kv[:, 1]
        else:
            qkv = qkv_r.reshape(B, 3, H, Lq, D)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        cfg = _FWD_CFG
        grid = (triton.cdiv(Lq, cfg["BM"]), B * H)
        inner = H * D
        w2d = w_out.reshape(H * D, H * D).clone()
        A_x = torch.empty(B, Lq, inner, device=dev, dtype=torch.float32)
        _attn_fwd_kernel[grid](
            q, k, v, A, lse, A_x,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            B, H, D, Lq, Lk, inner, scale,
            XSA=xsa, CAUSAL=is_causal, EVEN_N=Lk % cfg["BN"] == 0,
            PREC=cfg["prec"],
            BLOCK_M=cfg["BM"], BLOCK_N=cfg["BN"], BLOCK_D=cfg["BD"],
            num_warps=cfg["warp"], num_stages=cfg["stages"],
        )
        cfg = _OUT_CFG
        dim = H * D
        grid = (triton.cdiv(Lq, cfg["BM"]), triton.cdiv(dim, cfg["BN"]), B)
        _outproj_kernel[grid](
            A_x, w2d, bias, iden, out, B, Lq, inner, dim,
            HAS_BIAS=b_out is not None,
            PREC=cfg["prec"],
            BLOCK_M=cfg["BM"], BLOCK_N=cfg["BN"], BLOCK_K=cfg["BK"],
            num_warps=cfg["warp"], num_stages=cfg["stages"],
        )
        ctx.save_for_backward(qkv_r, kv_r, iden, w2d, b_out, A, lse)
        ctx.qkv_shape = q_in.shape
        ctx.kv_shape = kv_in.shape if is_cross else None
        ctx.plan = plan
        ctx.is_cross = is_cross
        ctx.xsa = xsa
        ctx.is_causal = is_causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q_in, kv_in, iden, w2d, b_out, A, lse = ctx.saved_tensors
        # q_in/kv_in here are the ROTATED tensors (as saved by forward)
        q_shape, kv_shape = ctx.qkv_shape, ctx.kv_shape
        dout = dout.contiguous()
        plan, is_cross, xsa, is_causal, scale = ctx.plan, ctx.is_cross, ctx.xsa, ctx.is_causal, ctx.scale
        B, H, D = plan["B"], plan["H"], plan["D"]
        Lq, Lk = plan["Lq"], plan["Lk"]
        dim = H * D
        dev = dout.device
        cfg = _BWD_CFG

        if is_cross:
            q = q_in[:, 0]
            kv = kv_in.reshape(B, 2, H, Lk, D)
            k, v = kv[:, 0], kv[:, 1]
            dqkv = torch.zeros(B, H, Lq, D, device=dev, dtype=q_in.dtype)
            dkv = torch.zeros(B, 2, H, Lk, D, device=dev, dtype=kv_in.dtype)
            dq_buf, dk_buf, dv_buf = dqkv, dkv[:, 0], dkv[:, 1]
        else:
            qkv = q_in.reshape(B, 3, H, Lq, D)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            dqkv = torch.zeros(B, 3, H, Lq, D, device=dev, dtype=q_in.dtype)
            dq_buf, dk_buf, dv_buf = dqkv[:, 0], dqkv[:, 1], dqkv[:, 2]

        dWp = torch.zeros(B, dim, H, D, device=dev, dtype=torch.float32)
        dbias = torch.zeros(dim, device=dev, dtype=torch.float32)
        cfg = _BWD_CFG
        grid = (triton.cdiv(Lq, cfg["BM"]), B * H)
        if Lq * Lk >= 1024 * 1024:
            dA_pre = torch.empty(B, H, Lq, D, device=dev, dtype=torch.float32)
            _bwd_prep_kernel[grid](
                A, lse, dout, v, dWp, dbias, dA_pre, dv_buf,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dv_buf.stride(0), dv_buf.stride(1), dv_buf.stride(2), dv_buf.stride(3),
                w2d,
                B, H, D, Lq, Lk, dim,
                XSA=xsa,
                PREC=cfg["prec"],
                BLOCK_M=cfg["BM"], BLOCK_D=cfg["BD"], CHUNK_C=cfg["CC"],
                num_warps=cfg["warp"], num_stages=1,
            )
            _attn_bwd_kernel[grid](
                q, k, v, dA_pre, lse, dq_buf, dk_buf, dv_buf,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dq_buf.stride(0), dq_buf.stride(1), dq_buf.stride(2), dq_buf.stride(3),
                dk_buf.stride(0), dk_buf.stride(1), dk_buf.stride(2), dk_buf.stride(3),
                dv_buf.stride(0), dv_buf.stride(1), dv_buf.stride(2), dv_buf.stride(3),
                B, H, D, Lq, Lk, scale,
                XSA=xsa, CAUSAL=is_causal, EVEN_N=Lk % cfg["BN"] == 0,
                PREC=cfg["prec"],
                BLOCK_M=cfg["BM"], BLOCK_N=cfg["BN"], BLOCK_D=cfg["BD"],
                num_warps=cfg["warp"], num_stages=1,
            )
        else:
            cfg_s = dict(BM=32, BN=64, BD=64, CC=64, prec="tf32", warp=4, stages=1)
            _attn_bwd_kernel_single[grid](
                q, k, v, A, lse, dout, dWp, dbias, dq_buf, dk_buf, dv_buf,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dq_buf.stride(0), dq_buf.stride(1), dq_buf.stride(2), dq_buf.stride(3),
                dk_buf.stride(0), dk_buf.stride(1), dk_buf.stride(2), dk_buf.stride(3),
                dv_buf.stride(0), dv_buf.stride(1), dv_buf.stride(2), dv_buf.stride(3),
                w2d,
                B, H, D, Lq, Lk, dim, scale,
                XSA=xsa, CAUSAL=is_causal, EVEN_N=Lk % cfg_s["BN"] == 0,
                PREC=cfg_s["prec"],
                BLOCK_M=cfg_s["BM"], BLOCK_N=cfg_s["BN"], BLOCK_D=cfg_s["BD"], CHUNK_C=cfg_s["CC"],
                num_warps=cfg_s["warp"], num_stages=cfg_s["stages"],
            )
        dW_out = torch.sum(dWp, dim=0).reshape(dim, dim, 1, 1)

        if plan["rotate"]:
            cfg = _ROT_CFG
            q_ = plan["q"]
            if is_cross:
                dq_grad = torch.empty(B, H, D, Lq, device=dev, dtype=q_in.dtype).reshape(q_shape)
                R = B * H * Lq
                grid = (triton.cdiv(R, cfg["BR"]), triton.cdiv(plan["A"] * q_, cfg["BI"]))
                _rot_kernel[grid](
                    dq_buf, dq_grad, B, H, D, Lq, plan["sp1"], plan["sp2"],
                    _tab(plan["cos0"], dev), _tab(plan["sin0"], dev),
                    _tab(plan["cos1"], dev), _tab(plan["sin1"], dev),
                    _tab(plan["cos2"], dev), _tab(plan["sin2"], dev),
                    q_, R, P=1, NP=1, A=plan["A"], INVERSE=True,
                    BLOCK_R=cfg["BR"], BLOCK_I=cfg["BI"], num_warps=cfg["warp"],
                )
                dkv_grad = torch.empty(B, 2, H, D, Lk, device=dev, dtype=kv_in.dtype).reshape(kv_shape)
                R = B * H * Lk
                grid = (triton.cdiv(R, cfg["BR"]), triton.cdiv(plan["A"] * q_, cfg["BI"]))
                _rot_kernel[grid](
                    dkv, dkv_grad, B, H, D, Lk, plan["sp1k"], plan["sp2k"],
                    _tab(plan["cos0k"], dev), _tab(plan["sin0k"], dev),
                    _tab(plan["cos1k"], dev), _tab(plan["sin1k"], dev),
                    _tab(plan["cos2k"], dev), _tab(plan["sin2k"], dev),
                    q_, R, P=2, NP=1, A=plan["A"], INVERSE=True,
                    BLOCK_R=cfg["BR"], BLOCK_I=cfg["BI"], num_warps=cfg["warp"],
                )
                dkv_grad.reshape(B, 2, H, D, Lk)[:, 1] = dkv[:, 1].transpose(-1, -2)
                return dq_grad, dkv_grad, dout, dW_out, dbias if b_out is not None else None, None, None, None, None, None
            else:
                dqkv_grad = torch.empty(B, 3, H, D, Lq, device=dev, dtype=q_in.dtype).reshape(q_shape)
                R = B * 2 * H * Lq
                grid = (triton.cdiv(R, cfg["BR"]), triton.cdiv(plan["A"] * q_, cfg["BI"]))
                _rot_kernel[grid](
                    dqkv, dqkv_grad, B, H, D, Lq, plan["sp1"], plan["sp2"],
                    _tab(plan["cos0"], dev), _tab(plan["sin0"], dev),
                    _tab(plan["cos1"], dev), _tab(plan["sin1"], dev),
                    _tab(plan["cos2"], dev), _tab(plan["sin2"], dev),
                    q_, R, P=3, NP=2, A=plan["A"], INVERSE=True,
                    BLOCK_R=cfg["BR"], BLOCK_I=cfg["BI"], num_warps=cfg["warp"],
                )
                dqkv_grad.reshape(B, 3, H, D, Lq)[:, 2] = dqkv[:, 2].transpose(-1, -2)
        elif is_cross:
            dqkv_grad = dqkv.reshape(q_shape)
            dkv_grad = dkv.reshape(kv_shape)
            return dqkv_grad, dkv_grad, dout, dW_out, dbias if b_out is not None else None, None, None, None, None, None
        else:
            dqkv_grad = dqkv.reshape(q_shape)

        return dqkv_grad, None, dout, dW_out, dbias if b_out is not None else None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Tuning configs
# ---------------------------------------------------------------------------

_FWD_CFG = dict(BM=64, BN=64, BD=64, prec="tf32", warp=4, stages=2)
_OUT_CFG = dict(BM=64, BN=64, BK=128, prec="tf32", warp=4, stages=2)
_BWD_CFG = dict(BM=64, BN=32, BD=64, CC=64, prec="tf32", warp=4, stages=1)
_ROT_CFG = dict(BR=16, BI=64, warp=8)

def _set_bd(cfg, D):
    cfg["BD"] = triton.next_power_of_2(D)


# ---------------------------------------------------------------------------
# Drop-in modules (same __init__ interface + state_dict as originals)
# ---------------------------------------------------------------------------

class FastSelfAttention(SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fused_cache = {}

    @torch.compiler.disable
    def _fused_plan(self, x):
        key = (tuple(x.shape), self.training)
        if key in self._fused_cache:
            return self._fused_cache[key]
        sp = tuple(x.shape[2:])
        A = len(sp)
        D = self.head_dim
        if A == 1:
            q = D // 2
            sin, cos = self.rotary_emb.get_1d_freq(x, self.rotary_emb.base, sp[0], q)
            cos0, sin0 = cos.reshape(sp[0], q), sin.reshape(sp[0], q)
            cos1 = sin1 = cos2 = sin2 = None
            sp1 = sp2 = 1
        elif A == 2:
            q = D // 4
            sin_h, cos_h, sin_w, cos_w = self.rotary_emb.get_2d_freqs(x, self.rotary_emb.base, sp[0], sp[1], q)
            cos0, sin0 = cos_h.reshape(sp[0], q), sin_h.reshape(sp[0], q)
            cos1, sin1 = cos_w.reshape(sp[1], q), sin_w.reshape(sp[1], q)
            cos2 = sin2 = None
            sp1, sp2 = sp[1], 1
        elif A == 3:
            q = D // 6
            sin_h, cos_h, sin_w, cos_w, sin_d, cos_d = self.rotary_emb.get_3d_freqs(
                x, self.rotary_emb.base, sp[0], sp[1], sp[2], q)
            cos0, sin0 = cos_h.reshape(sp[0], q), sin_h.reshape(sp[0], q)
            cos1, sin1 = cos_w.reshape(sp[1], q), sin_w.reshape(sp[1], q)
            cos2, sin2 = cos_d.reshape(sp[2], q), sin_d.reshape(sp[2], q)
            sp1, sp2 = sp[1], sp[2]
        else:
            raise RuntimeError("dimensions must be 1, 2 or 3")
        L = x[0].numel() // x.shape[1]
        if not self.add_rotary_embedding:
            cos0 = sin0 = cos1 = sin1 = cos2 = sin2 = None
        plan = dict(B=x.shape[0], H=self.heads, D=D, Lq=L, Lk=L, q=q, A=A,
                    cos0=cos0, sin0=sin0, cos1=cos1, sin1=sin1, cos2=cos2, sin2=sin2,
                    sp1=sp1, sp2=sp2, rotate=self.add_rotary_embedding)
        self._fused_cache[key] = plan
        return plan
    @torch.compiler.disable
    def forward(self, x):
        if x.device.type != "cuda" or self.head_dim not in (32, 64, 128) or not torch.cuda.is_available():
            return super().forward(x)
        if self.dropout > 0 and self.training:
            return super().forward(x)
        L = x[0].numel() // x.shape[1]
        if L < 16:
            return super().forward(x)
        iden = x
        x = self.norm(self.abs_emb(x))
        qkv = self.to_qkv(x)
        plan = self._fused_plan(x)
        out = _FusedAttnFn.apply(qkv, None, iden, self.to_out.weight, self.to_out.bias, plan,
                                 False, self.xsa, self.is_causal,
                                 1.0 / math.sqrt(self.head_dim))
        return out


class FastCrossAttention(CrossAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fused_cache = {}

    @torch.compiler.disable
    def _fused_plan(self, x, mem):
        key = (tuple(x.shape), tuple(mem.shape), self.training)
        if key in self._fused_cache:
            return self._fused_cache[key]
        spx, spm = tuple(x.shape[2:]), tuple(mem.shape[2:])
        A = len(spx)
        D = self.head_dim
        if A == 1:
            q = D // 2
            sin, cos = self.rotary_emb.get_1d_freq(x, self.rotary_emb.base, spx[0], q)
            cos0, sin0 = cos.reshape(spx[0], q), sin.reshape(spx[0], q)
            sin_k, cos_k = self.rotary_emb.get_1d_freq(mem, self.rotary_emb.base, spm[0], q)
            cos0k, sin0k = cos_k.reshape(spm[0], q), sin_k.reshape(spm[0], q)
            cos1 = sin1 = cos2 = sin2 = None
            cos1k = sin1k = cos2k = sin2k = None
            sp1 = sp2 = sp1k = sp2k = 1
        elif A == 2:
            q = D // 4
            sin_h, cos_h, sin_w, cos_w = self.rotary_emb.get_2d_freqs(x, self.rotary_emb.base, spx[0], spx[1], q)
            cos0, sin0 = cos_h.reshape(spx[0], q), sin_h.reshape(spx[0], q)
            cos1, sin1 = cos_w.reshape(spx[1], q), sin_w.reshape(spx[1], q)
            sin_kh, cos_kh, sin_kw, cos_kw = self.rotary_emb.get_2d_freqs(mem, self.rotary_emb.base, spm[0], spm[1], q)
            cos0k, sin0k = cos_kh.reshape(spm[0], q), sin_kh.reshape(spm[0], q)
            cos1k, sin1k = cos_kw.reshape(spm[1], q), sin_kw.reshape(spm[1], q)
            cos2 = sin2 = None
            cos2k = sin2k = None
            sp1, sp2 = spx[1], 1
            sp1k, sp2k = spm[1], 1
        elif A == 3:
            q = D // 6
            sin_h, cos_h, sin_w, cos_w, sin_d, cos_d = self.rotary_emb.get_3d_freqs(
                x, self.rotary_emb.base, spx[0], spx[1], spx[2], q)
            cos0, sin0 = cos_h.reshape(spx[0], q), sin_h.reshape(spx[0], q)
            cos1, sin1 = cos_w.reshape(spx[1], q), sin_w.reshape(spx[1], q)
            cos2, sin2 = cos_d.reshape(spx[2], q), sin_d.reshape(spx[2], q)
            sin_kh, cos_kh, sin_kw, cos_kw, sin_kd, cos_kd = self.rotary_emb.get_3d_freqs(
                mem, self.rotary_emb.base, spm[0], spm[1], spm[2], q)
            cos0k, sin0k = cos_kh.reshape(spm[0], q), sin_kh.reshape(spm[0], q)
            cos1k, sin1k = cos_kw.reshape(spm[1], q), sin_kw.reshape(spm[1], q)
            cos2k, sin2k = cos_kd.reshape(spm[2], q), sin_kd.reshape(spm[2], q)
            sp1, sp2 = spx[1], spx[2]
            sp1k, sp2k = spm[1], spm[2]
        else:
            raise RuntimeError("dimensions must be 1, 2 or 3")
        if not self.add_rotary_embedding:
            cos0 = sin0 = cos1 = sin1 = cos2 = sin2 = None
            cos0k = sin0k = cos1k = sin1k = cos2k = sin2k = None
        plan = dict(B=x.shape[0], H=self.heads, D=D,
                    Lq=x[0].numel() // x.shape[1], Lk=mem[0].numel() // mem.shape[1],
                    q=q, A=A,
                    cos0=cos0, sin0=sin0, cos1=cos1, sin1=sin1, cos2=cos2, sin2=sin2,
                    cos0k=cos0k, sin0k=sin0k, cos1k=cos1k, sin1k=sin1k, cos2k=cos2k, sin2k=sin2k,
                    sp1=sp1, sp2=sp2, sp1k=sp1k, sp2k=sp2k, rotate=self.add_rotary_embedding)
        self._fused_cache[key] = plan
        return plan

    @torch.compiler.disable
    def forward(self, x, memory):
        if x.device.type != "cuda" or self.head_dim not in (32, 64, 128) or not torch.cuda.is_available():
            return super().forward(x, memory)
        if self.dropout > 0 and self.training:
            return super().forward(x, memory)
        if self.is_causal:
            return super().forward(x, memory)
        Lq = x[0].numel() // x.shape[1]
        Lk = memory[0].numel() // memory.shape[1]
        if Lq < 16 or Lk < 16:
            return super().forward(x, memory)
        if self.xsa and Lq != Lk:
            return super().forward(x, memory)
        iden = x
        x = self.norm(self.x_abs_emb(x))
        memory = self.norm_context(self.mem_abs_emb(memory))
        q = self.to_q(x)
        kv = self.to_kv(memory)
        plan = self._fused_plan(x, memory)
        out = _FusedAttnFn.apply(q, kv, iden, self.to_out.weight, self.to_out.bias, plan,
                                 True, self.xsa, False,
                                 1.0 / math.sqrt(self.head_dim))
        return out
