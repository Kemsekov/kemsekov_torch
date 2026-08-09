"""Fused Triton kernels + custom autograd for AR2 semantics (see AR2Fast).

Semantics identical to AttentionResidual2 (production etalon):
  - per step i: k_i = l2norm(KV(xt_i)); v_i = xt_i
  - attention over keys 0..i (INCLUDING current) with query[i], scale = 1/d
  - softmax over the key axis, weighted sum of values -> out module -> module m
  - final key is NOT l2-normalized (etalon quirk), final attention over all n+1 keys
No in-place buffer hacks -> no memory leak.

Optimizations over the etalon:
  - KV (rmsnorm+silu+linear+l2norm) fused into 1 kernel (fwd + bwd)
  - attention (logits+softmax+weighted sum) fused into 1 kernel (fwd + bwd)
  - out (rmsnorm+silu+linear+bias+residual) fused into 1 kernel (fwd + bwd)
  - images (features_dimension=1): out writes directly into a channels_last
    [B,C,H,W] buffer (flat layout == [P,d]), so per-step reshape/transpose
    copies are eliminated and conv2d runs on NHWC data (faster on cudnn)
  - gradient weight matrices use staged reductions (scratch + torch.sum)
    instead of heavy atomic contention where it wins
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_EPS_BY_DTYPE = {torch.float32: 2.0 ** -23, torch.bfloat16: 2.0 ** -8}


def _kernel_cfg(d):
    bd = max(16, 1 << (d - 1).bit_length())
    w = 8 if bd >= 128 else 4
    return bd, w


def _use_fused(d):
    return d in (16, 32, 64)


def _kv_torch(x, w_rms, W, normalize):
    k = F.linear(F.silu(F.rms_norm(x, (x.shape[-1],), w_rms, None)), W)
    if normalize:
        k = F.normalize(k, 2.0, -1)
    return k


def _out_torch(S, w, b, W, alpha):
    h0 = F.rms_norm(S, (S.shape[-1],), w, None)
    return alpha * F.linear(F.silu(h0), W, b) + h0


# --------------------------------------------------------------------------
# Fused KV: rmsnorm -> silu -> linear -> (optional l2-normalize)
# --------------------------------------------------------------------------
@triton.jit
def kv_fwd_kernel(
    x_ptr, w_rms_ptr, W_ptr, k_ptr, v_ptr,
    P, d, row,
    eps, L2NORM: tl.constexpr, BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < P

    x_ptrs = x_ptr + offs_m[:, None] * d + offs_d[None, :]
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    x2 = x * x
    ms = tl.sum(x2, axis=1) / d
    rstd = 1.0 / tl.sqrt(ms + eps)
    w_rms = tl.load(w_rms_ptr + offs_d).to(tl.float32)
    h0 = x * rstd[:, None] * w_rms[None, :]
    if BF16:
        h0 = h0.to(tl.bfloat16).to(tl.float32)
    h = tl.sigmoid(h0) * h0                       # silu
    if BF16:
        h = h.to(tl.bfloat16).to(tl.float32)

    W = tl.load(W_ptr + offs_d[:, None] * d + offs_d[None, :]).to(tl.float32)  # [D,D]
    acc = tl.dot(h, tl.trans(W), input_precision="ieee")                  # y = h @ W^T
    if BF16:
        acc = acc.to(tl.bfloat16).to(tl.float32)
    if L2NORM:
        n2 = tl.sum(acc * acc, axis=1)
        denom = tl.maximum(tl.sqrt(tl.maximum(n2, 1e-24)), 1e-12)
        acc = acc / denom[:, None]
    k_ptrs = k_ptr + row * P * d + offs_m[:, None] * d + offs_d[None, :]
    tl.store(k_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None])
    v_ptrs = v_ptr + row * P * d + offs_m[:, None] * d + offs_d[None, :]
    tl.store(v_ptrs, x.to(v_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def kv_bwd_kernel(
    x_ptr, w_rms_ptr, W_ptr, dk_ptr, dx_ptr,
    dWp_ptr, dwrms_ptr,
    P, d, nb,
    eps, L2NORM: tl.constexpr, BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < P

    x_ptrs = x_ptr + offs_m[:, None] * d + offs_d[None, :]
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    x2 = x * x
    ms = tl.sum(x2, axis=1) / d
    rstd = 1.0 / tl.sqrt(ms + eps)
    w_rms = tl.load(w_rms_ptr + offs_d).to(tl.float32)
    h0 = x * rstd[:, None] * w_rms[None, :]
    if BF16:
        h0 = h0.to(tl.bfloat16).to(tl.float32)
    h = tl.sigmoid(h0) * h0
    if BF16:
        h = h.to(tl.bfloat16).to(tl.float32)

    W = tl.load(W_ptr + offs_d[:, None] * d + offs_d[None, :]).to(tl.float32)
    acc = tl.dot(h, tl.trans(W), input_precision="ieee")
    if L2NORM:
        n2 = tl.sum(acc * acc, axis=1)
        denom = tl.maximum(tl.sqrt(tl.maximum(n2, 1e-24)), 1e-12)
        k = acc / denom[:, None]
    else:
        k = acc

    dk = tl.load(dk_ptr + offs_m[:, None] * d + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    # l2norm backward: dacc = (dk - k*dot(dk,k)) / n   (identity if L2NORM=0)
    if L2NORM:
        dk_dot_k = tl.sum(dk * k, axis=1)
        dacc = (dk - k * dk_dot_k[:, None]) / denom[:, None]
    else:
        dacc = dk
    # silu backward
    dh0 = tl.dot(dacc, W, input_precision="ieee")
    sig = tl.sigmoid(h0)
    dh0 = dh0 * sig * (1.0 + h0 * (1.0 - sig))
    # linear weight grad: dW = dacc^T @ h  (staged: write partial, torch sums)
    dW_part = tl.dot(tl.trans(dacc), h, input_precision="ieee")
    tl.store(dWp_ptr + pid * d * d + offs_d[:, None] * d + offs_d[None, :],
             dW_part.to(dWp_ptr.dtype.element_ty))
    # rmsnorm weight grad: dw_rms[d] = sum_p u * dh0, u = x*rstd = h0/w_rms
    u = h0 / w_rms[None, :]
    dwrms_part = tl.sum(u * dh0, axis=0)
    tl.atomic_add(dwrms_ptr + offs_d, dwrms_part.to(dwrms_ptr.dtype.element_ty), sem="relaxed")
    # rmsnorm input grad: dx = rstd * (w*dh0 - u*mean(h0*dh0))
    hdh = tl.sum(h0 * dh0, axis=1) / d
    dx = rstd[:, None] * (dh0 * w_rms[None, :] - u * hdh[:, None])
    tl.store(dx_ptr + offs_m[:, None] * d + offs_d[None, :], dx.to(x_ptr.dtype.element_ty), mask=mask_m[:, None])


class _KVFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_rms, W, K_buf, V_buf, row, normalize=True):
        P, d = x.shape
        grid = (triton.cdiv(P, 64),)
        bd, nw = _kernel_cfg(d)
        kv_fwd_kernel[grid](x, w_rms, W, K_buf, V_buf, P, d, row,
                            _EPS_BY_DTYPE[x.dtype], bool(normalize),
                            x.dtype == torch.bfloat16, BLOCK_M=64, BLOCK_D=bd, num_warps=nw, num_stages=1)
        ctx.save_for_backward(x, w_rms, W)
        ctx.normalize = bool(normalize)
        return K_buf[row]

    @staticmethod
    def backward(ctx, dk):
        x, w_rms, W = ctx.saved_tensors
        P, d = x.shape
        dx = torch.empty_like(x)
        nb = triton.cdiv(P, 64)
        dWp = torch.empty(nb, d, d, device=x.device, dtype=torch.float32)
        dW = torch.zeros(d, d, device=x.device, dtype=torch.float32)
        dwrms = torch.zeros(d, device=x.device, dtype=torch.float32)
        grid = (nb,)
        bd, nw = _kernel_cfg(d)
        kv_bwd_kernel[grid](x, w_rms, W, dk, dx, dWp, dwrms, P, d, nb,
                            _EPS_BY_DTYPE[x.dtype], ctx.normalize, x.dtype == torch.bfloat16,
                            BLOCK_M=64, BLOCK_D=bd, num_warps=nw, num_stages=1)
        torch.sum(dWp, 0, out=dW)
        return dx, dwrms, dW, None, None, None, None


# --------------------------------------------------------------------------
# Fused attention: logits = (K*q)/d, softmax over keys, weighted sum of V
# --------------------------------------------------------------------------
@triton.jit
def attn_fwd_kernel(
    K_ptr, V_ptr, q_ptr, S_ptr, l_ptr,
    P, d, nk,
    scale, BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < P

    mask_d = offs_d < d
    colmask = mask_m[:, None] & mask_d[None, :]
    q = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    z = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for j in range(0, nk):
        k_ptrs = K_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :]
        k = tl.load(k_ptrs, mask=colmask, other=0.0).to(tl.float32)
        v_ptrs = V_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :]
        v = tl.load(v_ptrs, mask=colmask, other=0.0).to(tl.float32)
        p = k * q[None, :]
        if BF16:
            p = p.to(tl.bfloat16).to(tl.float32)
        l = tl.sum(p, axis=1) * scale
        if BF16:
            l = l.to(tl.bfloat16).to(tl.float32)
        tl.store(l_ptr + j * P + offs_m, l.to(l_ptr.dtype.element_ty), mask=mask_m)
        m_new = tl.maximum(m, l)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(l - m_new)
        vb = v * beta[:, None]
        if BF16:
            vb = vb.to(tl.bfloat16).to(tl.float32)
        acc = acc * alpha[:, None] + vb
        z = z * alpha + beta
        m = m_new
    acc = acc / z[:, None]
    out_ptrs = S_ptr + offs_m[:, None] * d + offs_d[None, :]
    tl.store(out_ptrs, acc.to(S_ptr.dtype.element_ty), mask=colmask)


@triton.jit
def attn_bwd_kernel(
    K_ptr, V_ptr, q_ptr, dS_ptr, dq_ptr,
    dK_ptr, dV_ptr, l_ptr,
    P, d, nk,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < P

    mask_d = offs_d < d
    colmask = mask_m[:, None] & mask_d[None, :]
    q = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    dS = tl.load(dS_ptr + offs_m[:, None] * d + offs_d[None, :], mask=colmask, other=0.0).to(tl.float32)

    # pass 1: softmax normalizer from saved logits
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    z = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for j in range(0, nk):
        l_j = tl.load(l_ptr + j * P + offs_m, mask=mask_m, other=0.0)
        m_new = tl.maximum(m, l_j)
        z = z * tl.exp(m - m_new) + tl.exp(l_j - m_new)
        m = m_new
    # pass 2: total = sum_j w_j * (dS . v_j)
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for j in range(0, nk):
        l_j = tl.load(l_ptr + j * P + offs_m, mask=mask_m, other=0.0)
        v = tl.load(V_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :], mask=colmask, other=0.0).to(tl.float32)
        w_j = tl.exp(l_j - m) / z
        total += w_j * tl.sum(dS * v, axis=1)
    # pass 3: dl_j = w_j*(c_j-total); dK_j = dl_j*q; dV_j = dS*w_j; dq += dl_j*k_j
    dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in range(0, nk):
        l_j = tl.load(l_ptr + j * P + offs_m, mask=mask_m, other=0.0)
        k = tl.load(K_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :], mask=colmask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :], mask=colmask, other=0.0).to(tl.float32)
        w_j = tl.exp(l_j - m) / z
        dl = w_j * (tl.sum(dS * v, axis=1) - total)
        dq_acc += tl.sum(dl[:, None] * k, axis=0)
        tl.store(dK_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :],
                 (dl[:, None] * q[None, :] * scale).to(dK_ptr.dtype.element_ty), mask=colmask)
        tl.store(dV_ptr + j * P * d + offs_m[:, None] * d + offs_d[None, :],
                 (dS * w_j[:, None]).to(dV_ptr.dtype.element_ty), mask=colmask)
    tl.atomic_add(dq_ptr + offs_d, (dq_acc * scale).to(dq_ptr.dtype.element_ty), sem="relaxed")


class _AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, K_buf, V_buf, nk, *kvs):
        # kvs = [k_0, v_0, k_1, v_1, ..., k_{nk-1}, v_{nk-1}] — buffer-row views,
        # kept as inputs only for gradient tracking; kernels read the buffers.
        P, d = K_buf.shape[1], K_buf.shape[2]
        S = torch.empty(P, d, device=K_buf.device, dtype=K_buf.dtype)
        l = torch.empty(nk, P, device=K_buf.device, dtype=K_buf.dtype)
        grid = (triton.cdiv(P, 64),)
        bd, nw = _kernel_cfg(d)
        attn_fwd_kernel[grid](K_buf, V_buf, q, S, l, P, d, nk, 1.0 / d,
                              K_buf.dtype == torch.bfloat16, BLOCK_M=64, BLOCK_D=bd, num_warps=nw, num_stages=1)
        ctx.save_for_backward(q, K_buf, V_buf, *kvs, l)
        ctx.nk = nk
        return S

    @staticmethod
    def backward(ctx, dS):
        saved = ctx.saved_tensors
        q, K_buf, V_buf = saved[0], saved[1], saved[2]
        nk = ctx.nk
        l = saved[-1]
        P, d = K_buf.shape[1], K_buf.shape[2]
        dK = torch.empty_like(K_buf[:nk])
        dV = torch.empty_like(V_buf[:nk])
        dq = torch.zeros(d, device=K_buf.device, dtype=torch.float32)
        grid = (triton.cdiv(P, 64),)
        bd, nw = _kernel_cfg(d)
        attn_bwd_kernel[grid](K_buf, V_buf, q, dS, dq, dK, dV, l, P, d, nk, 1.0 / d,
                              BLOCK_M=64, BLOCK_D=bd, num_warps=nw, num_stages=1)
        grads = []
        for j in range(nk):
            grads.append(dK[j])
            grads.append(dV[j])
        return (dq, None, None, None, *grads)


# --------------------------------------------------------------------------
# Fused out: rmsnorm -> silu -> linear (+bias) -> alpha*residual + skip
# --------------------------------------------------------------------------
@triton.jit
def out_fwd_kernel(
    S_ptr, w_ptr, b_ptr, W_ptr, out_ptr,
    P, d,
    alpha_ptr, eps, BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < P

    s_ptrs = S_ptr + offs_m[:, None] * d + offs_d[None, :]
    s = tl.load(s_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    ms = tl.sum(s * s, axis=1) / d
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    h0 = s * rstd[:, None] * w[None, :]
    if BF16:
        h0 = h0.to(tl.bfloat16).to(tl.float32)
    h = tl.sigmoid(h0) * h0
    if BF16:
        h = h.to(tl.bfloat16).to(tl.float32)

    W = tl.load(W_ptr + offs_d[:, None] * d + offs_d[None, :]).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)
    alpha = tl.load(alpha_ptr).to(tl.float32)
    m = tl.dot(h, tl.trans(W), input_precision="ieee") + b[None, :]
    if BF16:
        m = m.to(tl.bfloat16).to(tl.float32)
    y = alpha * m + h0
    out_ptrs = out_ptr + offs_m[:, None] * d + offs_d[None, :]
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def out_bwd_kernel(
    S_ptr, w_ptr, b_ptr, W_ptr, dout_ptr, dS_ptr,
    dWp_ptr, db_ptr, dw_ptr, dalpha_ptr,
    P, d, nb,
    alpha_ptr, eps,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < P

    s_ptrs = S_ptr + offs_m[:, None] * d + offs_d[None, :]
    s = tl.load(s_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    ms = tl.sum(s * s, axis=1) / d
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    h0 = s * rstd[:, None] * w[None, :]
    h = tl.sigmoid(h0) * h0

    W = tl.load(W_ptr + offs_d[:, None] * d + offs_d[None, :]).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)
    m = tl.dot(h, tl.trans(W), input_precision="ieee") + b[None, :]

    dout = tl.load(dout_ptr + offs_m[:, None] * d + offs_d[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr).to(tl.float32)

    # linear backward: dL/dm = alpha*dout
    dm = alpha * dout
    dW_part = tl.dot(tl.trans(dm), h, input_precision="ieee")
    tl.store(dWp_ptr + pid * d * d + offs_d[:, None] * d + offs_d[None, :],
             dW_part.to(dWp_ptr.dtype.element_ty))
    db_part = tl.sum(dm, axis=0)
    tl.atomic_add(db_ptr + offs_d, db_part.to(db_ptr.dtype.element_ty), sem="relaxed")
    # dalpha = sum(m * dout)
    dalpha_part = tl.sum(m * dout, axis=0)
    tl.atomic_add(dalpha_ptr + tl.arange(0, 1), tl.sum(dalpha_part), sem="relaxed")
    # silu backward + skip connection: dL/dh0 = dout + silu'(h0)*(dm @ W)
    dh0 = tl.dot(dm, W, input_precision="ieee")
    sig = tl.sigmoid(h0)
    dh0 = dh0 * sig * (1.0 + h0 * (1.0 - sig)) + dout
    # rmsnorm weight grad: dw[d] = sum_p u * dh0, u = S*rstd = h0/w
    u = h0 / w[None, :]
    dw_part = tl.sum(u * dh0, axis=0)
    tl.atomic_add(dw_ptr + offs_d, dw_part.to(dw_ptr.dtype.element_ty), sem="relaxed")
    # rmsnorm backward: dS = rstd * (w*dh0 - u*mean(u*w*dh0))
    dy = dh0 * w[None, :]
    u_dy = tl.sum(u * dy, axis=1) / d
    dS = rstd[:, None] * (dy - u * u_dy[:, None])
    tl.store(dS_ptr + offs_m[:, None] * d + offs_d[None, :], dS.to(dS_ptr.dtype.element_ty), mask=mask_m[:, None])


class _OutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, w, b, W, alpha):
        P, d = S.shape
        out = torch.empty_like(S)
        grid = (triton.cdiv(P, 64),)
        bd, nw = _kernel_cfg(d)
        out_fwd_kernel[grid](S, w, b, W, out, P, d, alpha,
                             _EPS_BY_DTYPE[S.dtype], S.dtype == torch.bfloat16,
                             BLOCK_M=64, BLOCK_D=bd, num_warps=nw, num_stages=1)
        ctx.save_for_backward(S, w, b, W, alpha)
        return out

    @staticmethod
    def backward(ctx, dout):
        S, w, b, W, alpha = ctx.saved_tensors
        P, d = S.shape
        dS = torch.empty_like(S)
        nb = triton.cdiv(P, 64)
        dWp = torch.empty(nb, d, d, device=S.device, dtype=torch.float32)
        dW = torch.zeros(d, d, device=S.device, dtype=torch.float32)
        db = torch.zeros(d, device=S.device, dtype=torch.float32)
        dw = torch.zeros(d, device=S.device, dtype=torch.float32)
        dalpha = torch.zeros(1, device=S.device, dtype=torch.float32)
        grid = (nb,)
        bd, nw = _kernel_cfg(d)
        out_bwd_kernel[grid](S, w, b, W, dout, dS, dWp, db, dw, dalpha, P, d, nb,
                             alpha, _EPS_BY_DTYPE[S.dtype], BLOCK_M=64, BLOCK_D=bd, num_warps=nw, num_stages=1)
        torch.sum(dWp, 0, out=dW)
        return dS, dw, db, dW, dalpha


# --------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------
