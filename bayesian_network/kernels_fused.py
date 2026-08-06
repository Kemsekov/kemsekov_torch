"""
Persistent fully-fused out_prob MLP kernel.

ONE kernel, grid = number of resident blocks (saturates all SMs):
  - every program loads all 4 weight tiles + head weights ONCE (shared)
  - then loops over its share of row-tiles, computing the FULL MLP
    (2 residual blocks + head: 5 matmuls, 3 RMSNorms, activations)
    entirely in shared/registers -- no intermediate global traffic
  - writes only the final out vector

Backward: same persistent structure, per-block partial weight grads in
global scratch, then one tiny reduce kernel.
"""
import torch
import triton
import triton.language as tl

EPS = 1e-6


@triton.jit
def _load_wt(w_ptr, d, D: tl.constexpr):
    return tl.load(w_ptr + d[None, :] * D + d[:, None])


@triton.jit
def _outprob_fwd_kernel(
    xt_ptr, out_ptr,
    g1_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
    g2_ptr, w3_ptr, b3_ptr, w4_ptr, b4_ptr,
    g3_ptr, w5_ptr, b5_ptr,
    R, eps,
    NBLK: tl.constexpr, BLOCK_R: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, D)

    # --- load weights once (shared) ---
    W1t = _load_wt(w1_ptr, d, D)
    W2t = _load_wt(w2_ptr, d, D)
    W3t = _load_wt(w3_ptr, d, D)
    W4t = _load_wt(w4_ptr, d, D)
    b1 = tl.load(b1_ptr + d)
    b2 = tl.load(b2_ptr + d)
    b3 = tl.load(b3_ptr + d)
    b4 = tl.load(b4_ptr + d)
    g1 = tl.load(g1_ptr + d)
    g2 = tl.load(g2_ptr + d)
    g3 = tl.load(g3_ptr + d)
    w5 = tl.load(w5_ptr + d)
    b5 = tl.load(b5_ptr)

    n_tiles = tl.cdiv(R, BLOCK_R)
    for t in tl.range(pid, n_tiles, NBLK):
        rows = t * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < R
        h = tl.load(xt_ptr + rows[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)

        # residual block 1
        m1 = tl.sum(h * h, axis=1, keep_dims=True) / D
        r1 = tl.rsqrt(m1 + eps)
        n1 = h * r1 * g1[None, :]
        a1 = tl.dot(n1, W1t, input_precision="ieee") + b1[None, :]
        t1 = tl.extra.libdevice.tanh(a1)
        p1 = n1 * t1
        s1 = p1 * tl.sigmoid(p1)
        h = h + (tl.dot(s1, W2t, input_precision="ieee") + b2[None, :])

        # residual block 2
        m2 = tl.sum(h * h, axis=1, keep_dims=True) / D
        r2 = tl.rsqrt(m2 + eps)
        n2 = h * r2 * g2[None, :]
        a2 = tl.dot(n2, W3t, input_precision="ieee") + b3[None, :]
        t2 = tl.extra.libdevice.tanh(a2)
        p2 = n2 * t2
        s2 = p2 * tl.sigmoid(p2)
        h = h + (tl.dot(s2, W4t, input_precision="ieee") + b4[None, :])

        # head
        m3 = tl.sum(h * h, axis=1, keep_dims=True) / D
        r3 = tl.rsqrt(m3 + eps)
        n3 = h * r3 * g3[None, :]
        s3 = n3 * tl.sigmoid(n3)
        out = tl.sum(s3 * w5[None, :], axis=1) + b5

        tl.store(out_ptr + rows, out, mask=rmask)


@triton.jit
def _outprob_bwd_kernel(
    xt_ptr, dout_ptr, dxt_ptr,
    g1_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
    g2_ptr, w3_ptr, b3_ptr, w4_ptr, b4_ptr,
    g3_ptr, w5_ptr, b5_ptr,
    dg1_ptr, dw1_ptr, db1_ptr, dw2_ptr, db2_ptr,
    dg2_ptr, dw3_ptr, db3_ptr, dw4_ptr, db4_ptr,
    dg3_ptr, dw5_ptr, db5_ptr,
    R, eps,
    NBLK: tl.constexpr, BLOCK_R: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, D)

    W1t = _load_wt(w1_ptr, d, D)
    W2t = _load_wt(w2_ptr, d, D)
    W3t = _load_wt(w3_ptr, d, D)
    W4t = _load_wt(w4_ptr, d, D)
    b1 = tl.load(b1_ptr + d)
    b2 = tl.load(b2_ptr + d)
    b3 = tl.load(b3_ptr + d)
    b4 = tl.load(b4_ptr + d)
    g1 = tl.load(g1_ptr + d)
    g2 = tl.load(g2_ptr + d)
    g3 = tl.load(g3_ptr + d)
    w5 = tl.load(w5_ptr + d)

    n_tiles = tl.cdiv(R, BLOCK_R)
    for t in tl.range(pid, n_tiles, NBLK):
        rows = t * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < R
        h = tl.load(xt_ptr + rows[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
        dout = tl.load(dout_ptr + rows, mask=rmask, other=0.0)

        # ---- forward recompute ----
        m1 = tl.sum(h * h, axis=1, keep_dims=True) / D
        r1 = tl.rsqrt(m1 + eps)
        n1 = h * r1 * g1[None, :]
        a1 = tl.dot(n1, W1t, input_precision="ieee") + b1[None, :]
        t1 = tl.extra.libdevice.tanh(a1)
        p1 = n1 * t1
        s1 = p1 * tl.sigmoid(p1)
        u1 = tl.dot(s1, W2t, input_precision="ieee") + b2[None, :]
        h1 = h + u1

        m2 = tl.sum(h1 * h1, axis=1, keep_dims=True) / D
        r2 = tl.rsqrt(m2 + eps)
        n2 = h1 * r2 * g2[None, :]
        a2 = tl.dot(n2, W3t, input_precision="ieee") + b3[None, :]
        t2 = tl.extra.libdevice.tanh(a2)
        p2 = n2 * t2
        s2 = p2 * tl.sigmoid(p2)
        u2 = tl.dot(s2, W4t, input_precision="ieee") + b4[None, :]
        h2 = h1 + u2

        m3 = tl.sum(h2 * h2, axis=1, keep_dims=True) / D
        r3 = tl.rsqrt(m3 + eps)
        n3 = h2 * r3 * g3[None, :]
        s3 = n3 * tl.sigmoid(n3)

        # ---- backward ----
        ds3 = dout[:, None] * w5[None, :]
        sig3 = tl.sigmoid(n3)
        dn3 = ds3 * (sig3 + n3 * sig3 * (1.0 - sig3))
        dot3 = tl.sum(dn3 * h2 * g3[None, :], axis=1, keep_dims=True)
        dh2 = r3 * g3[None, :] * dn3 - (r3 * r3 * r3 / D) * h2 * dot3
        tl.store(dw5_ptr + pid * D + d, tl.sum(dout[:, None] * s3, axis=0))
        tl.store(db5_ptr + pid, tl.sum(dout, axis=0))
        tl.store(dg3_ptr + pid * D + d, tl.sum(dn3 * (h2 * r3), axis=0))

        du2 = dh2
        ds2 = tl.trans(tl.dot(W4t, tl.trans(du2), input_precision="ieee"))
        sig2 = tl.sigmoid(p2)
        dp2 = ds2 * (sig2 + p2 * sig2 * (1.0 - sig2))
        dn2 = dp2 * t2
        dt2 = dp2 * n2
        da2 = dt2 * (1.0 - t2 * t2)
        dn2 = dn2 + tl.trans(tl.dot(W3t, tl.trans(da2), input_precision="ieee"))
        tl.store(dw4_ptr + pid * D * D + d[:, None] * D + d[None, :],
                 tl.dot(tl.trans(du2), s2, input_precision="ieee"))
        tl.store(db4_ptr + pid * D + d, tl.sum(du2, axis=0))
        tl.store(dw3_ptr + pid * D * D + d[:, None] * D + d[None, :],
                 tl.dot(tl.trans(da2), n2, input_precision="ieee"))
        tl.store(db3_ptr + pid * D + d, tl.sum(da2, axis=0))
        dot2 = tl.sum(dn2 * h1 * g2[None, :], axis=1, keep_dims=True)
        dh1 = r2 * g2[None, :] * dn2 - (r2 * r2 * r2 / D) * h1 * dot2
        tl.store(dg2_ptr + pid * D + d, tl.sum(dn2 * (h1 * r2), axis=0))

        du1 = dh1
        ds1 = tl.trans(tl.dot(W2t, tl.trans(du1), input_precision="ieee"))
        sig1 = tl.sigmoid(p1)
        dp1 = ds1 * (sig1 + p1 * sig1 * (1.0 - sig1))
        dn1 = dp1 * t1
        dt1 = dp1 * n1
        da1 = dt1 * (1.0 - t1 * t1)
        dn1 = dn1 + tl.trans(tl.dot(W1t, tl.trans(da1), input_precision="ieee"))
        tl.store(dw2_ptr + pid * D * D + d[:, None] * D + d[None, :],
                 tl.dot(tl.trans(du1), s1, input_precision="ieee"))
        tl.store(db2_ptr + pid * D + d, tl.sum(du1, axis=0))
        tl.store(dw1_ptr + pid * D * D + d[:, None] * D + d[None, :],
                 tl.dot(tl.trans(da1), n1, input_precision="ieee"))
        tl.store(db1_ptr + pid * D + d, tl.sum(da1, axis=0))
        dot1 = tl.sum(dn1 * h * g1[None, :], axis=1, keep_dims=True)
        dh = r1 * g1[None, :] * dn1 - (r1 * r1 * r1 / D) * h * dot1
        tl.store(dg1_ptr + pid * D + d, tl.sum(dn1 * (h * r1), axis=0))

        tl.store(dxt_ptr + rows[:, None] * D + d[None, :],
                 dh + u1 * 0.0, mask=rmask[:, None])


@triton.jit
def _reduce_kernel(part_ptr, out_ptr, NBLK, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for b in range(NBLK):
        acc += tl.load(part_ptr + b * N + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, acc, mask=m)


def _num_blocks(BLOCK_R, num_warps, dev):
    """number of resident blocks: one wave saturating the GPU."""
    props = torch.cuda.get_device_properties(dev)
    # shared per block for this kernel (4 weight tiles + staging, BLOCK_R rows)
    # conservative: 4*D*D*4 + 3*BLOCK_R*D*4 bytes
    D = 64
    sh = 4 * D * D * 4 + 3 * BLOCK_R * D * 4 + 4096
    per_sm = max(1, (props.shared_memory_per_multiprocessor * 3 // 4) // sh)
    return props.multi_processor_count * per_sm


class OutProbFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xt, g1, w1, b1, w2, b2, g2, w3, b3, w4, b4, g3, w5, b5,
                BLOCK_R=64, num_warps=4, nblk=None):
        R, D = xt.shape
        xt = xt.contiguous()
        dev = xt.device
        out = torch.empty(R, device=dev, dtype=xt.dtype)
        if nblk is None:
            nblk = _num_blocks(BLOCK_R, num_warps, dev)
        grid = (nblk,)
        _outprob_fwd_kernel[grid](
            xt, out, g1, w1, b1, w2, b2, g2, w3, b3, w4, b4, g3, w5, b5,
            R, EPS, NBLK=nblk, BLOCK_R=BLOCK_R, D=D, num_stages=1, num_warps=num_warps)
        ctx.save_for_backward(xt, g1, w1, b1, w2, b2, g2, w3, b3, w4, b4, g3, w5, b5)
        ctx.BLOCK_R, ctx.num_warps, ctx.nblk = BLOCK_R, num_warps, nblk
        return out

    @staticmethod
    def backward(ctx, dout):
        xt, g1, w1, b1, w2, b2, g2, w3, b3, w4, b4, g3, w5, b5 = ctx.saved_tensors
        R, D = xt.shape
        dev = xt.device
        nb = ctx.nblk
        dxt = torch.empty_like(xt)
        dg1 = torch.zeros_like(g1); dw1 = torch.zeros_like(w1); db1 = torch.zeros_like(b1)
        dw2 = torch.zeros_like(w2); db2 = torch.zeros_like(b2)
        dg2 = torch.zeros_like(g2); dw3 = torch.zeros_like(w3); db3 = torch.zeros_like(b3)
        dw4 = torch.zeros_like(w4); db4 = torch.zeros_like(b4)
        dg3 = torch.zeros_like(g3); dw5 = torch.zeros_like(w5); db5 = torch.zeros_like(b5)
        dg1p = torch.zeros(nb, D, device=dev); dw1p = torch.zeros(nb, D, D, device=dev)
        db1p = torch.zeros(nb, D, device=dev); dw2p = torch.zeros(nb, D, D, device=dev)
        db2p = torch.zeros(nb, D, device=dev)
        dg2p = torch.zeros(nb, D, device=dev); dw3p = torch.zeros(nb, D, D, device=dev)
        db3p = torch.zeros(nb, D, device=dev); dw4p = torch.zeros(nb, D, D, device=dev)
        db4p = torch.zeros(nb, D, device=dev)
        dg3p = torch.zeros(nb, D, device=dev); dw5p = torch.zeros(nb, D, device=dev)
        db5p = torch.zeros(nb, device=dev)
        grid = (nb,)
        _outprob_bwd_kernel[grid](
            xt, dout, dxt,
            g1, w1, b1, w2, b2, g2, w3, b3, w4, b4, g3, w5, b5,
            dg1p, dw1p, db1p, dw2p, db2p, dg2p, dw3p, db3p, dw4p, db4p, dg3p, dw5p, db5p,
            R, EPS, NBLK=nb, BLOCK_R=ctx.BLOCK_R, D=D, num_stages=1, num_warps=ctx.num_warps)
        rg = (triton.cdiv(D, 256),)
        _reduce_kernel[rg](dg1p, dg1, nb, D, BLOCK=256)
        _reduce_kernel[rg](dg2p, dg2, nb, D, BLOCK=256)
        _reduce_kernel[rg](dg3p, dg3, nb, D, BLOCK=256)
        _reduce_kernel[rg](db1p, db1, nb, D, BLOCK=256)
        _reduce_kernel[rg](db2p, db2, nb, D, BLOCK=256)
        _reduce_kernel[rg](db3p, db3, nb, D, BLOCK=256)
        _reduce_kernel[rg](db4p, db4, nb, D, BLOCK=256)
        _reduce_kernel[rg](dw5p, dw5, nb, D, BLOCK=256)
        rw = (triton.cdiv(D * D, 256),)
        _reduce_kernel[rw](dw1p, dw1, nb, D * D, BLOCK=256)
        _reduce_kernel[rw](dw2p, dw2, nb, D * D, BLOCK=256)
        _reduce_kernel[rw](dw3p, dw3, nb, D * D, BLOCK=256)
        _reduce_kernel[rw](dw4p, dw4, nb, D * D, BLOCK=256)
        r1 = (triton.cdiv(nb, 256),)
        _reduce_kernel[r1](db5p, db5, nb, 1, BLOCK=256)
        return dxt, dg1, dw1, db1, dw2, db2, dg2, dw3, db3, dw4, db4, dg3, dw5, db5, None, None
