"""Compact scrambled-Sobol standard-normal sampler (same logic as sobol_normal.py).

GPU: single fused Triton kernel (Gray-code Sobol XOR + fp32 erfinv + antithetic
mirror) -- no CPU work, no device copies. CPU: same formula, vectorized with a
precomputed 2^16 XOR table. Stateless: each call draws from index 0.
Drop-in: sample_base(sobol, count, device) like the original flow_matching one.
"""

import math

import torch
import triton
import triton.language as tl
from torch.quasirandom import SobolEngine

_INV32 = tl.constexpr(1.0 / 4294967296.0)
_SQ2 = tl.constexpr(1.4142135623730951)
_cache = {}


@triton.jit
def _erfinv(p):
    pa = tl.abs(p)
    t = pa * pa
    num = (((((-7.04501846e-01 * t + 3.32708506e+00) * t + -4.19001103e+00) * t
            + 6.98902936e-01) * t + 8.86226996e-01))
    den = ((((((8.10231906e-02 * t + -1.48307168e+00) * t + 4.89343025e+00) * t
              + -5.00994502e+00) * t + 5.26837254e-01) * t + 1.0))
    zc = pa * num / den
    q = tl.sqrt(-2.0 * tl.log(1.0 - pa))
    w = 1.0 / (q * q)
    num_t = (((((3.75033255e+02 * w + 1.85445978e+03) * w + 7.92216881e+02) * w
               + 5.91726083e+01) * w + 7.05950790e-01))
    den_t = (((((1.55215414e+00 * w + 2.82453336e+03) * w + 4.23685894e+03) * w
               + 1.29393864e+03) * w + 8.67150756e+01) * w + 1.0)
    zt = q * num_t / den_t
    z = tl.where(pa <= 0.95, zc, zt)
    z = tl.where(p >= 0.0, z, -z)
    z = tl.where(p <= -1.0, float("-inf"), z)
    z = tl.where(p >= 1.0, float("inf"), z)
    return z


@triton.jit
def _kernel(dirs, shift, out, half,
            DIM: tl.constexpr, DIMP2: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = row < half
    g = (row.to(tl.uint32) ^ (row.to(tl.uint32) >> 1))
    cols = tl.arange(0, DIMP2)
    cm = cols < DIM
    sh = tl.load(shift + cols, mask=cm, other=0)
    acc = tl.zeros([BLOCK, DIMP2], tl.uint32)
    for k in tl.static_range(32):
        dk = tl.load(dirs + k * DIM + cols, mask=cm, other=0)
        acc ^= dk[None, :] * ((g >> k) & 1)[:, None]
    z = _erfinv((acc ^ sh[None, :]).to(tl.float32) * _INV32 * 2.0 - 1.0) * _SQ2
    off = row[:, None] * DIM + cols[None, :]
    mm = m[:, None] & cm[None, :]
    z = tl.where(mm, z, 0.0)
    tl.store(out + off, z, mask=mm)
    tl.store(out + half * DIM + off, -z, mask=mm)


def _erfinv_cpu(p):
    pa = torch.abs(p)
    t = pa * pa
    num = ((((-7.04501846e-01 * t + 3.32708506e+00) * t + -4.19001103e+00) * t
            + 6.98902936e-01) * t + 8.86226996e-01)
    den = (((((8.10231906e-02 * t + -1.48307168e+00) * t + 4.89343025e+00) * t
             + -5.00994502e+00) * t + 5.26837254e-01) * t + 1.0)
    zc = pa * num / den
    q = torch.sqrt(-2.0 * torch.log(1.0 - pa))
    w = 1.0 / (q * q)
    num_t = (((((3.75033255e+02 * w + 1.85445978e+03) * w + 7.92216881e+02) * w
               + 5.91726083e+01) * w + 7.05950790e-01))
    den_t = (((((1.55215414e+00 * w + 2.82453336e+03) * w + 4.23685894e+03) * w
               + 1.29393864e+03) * w + 8.67150756e+01) * w + 1.0)
    zt = q * num_t / den_t
    z = torch.where(pa <= 0.95, zc, zt)
    z = torch.where(p >= 0.0, z, -z)
    z = torch.where(p <= -1.0, -float("inf"), z)
    z = torch.where(p >= 1.0, float("inf"), z)
    return z


def _cpu(out, d, sh, tbl, half, dim):
    gb = torch.arange(65536, dtype=torch.long) ^ (torch.arange(65536, dtype=torch.long) >> 1)
    for s in range(0, half, 65536):
        n = min(65536, half - s)
        g = gb[:n] ^ (((s >> 16) & 1) << 15)
        gh = (s >> 16) ^ (s >> 17)
        xh = torch.zeros((1, dim), dtype=torch.long)
        for k in range(16, 32):
            if (gh >> (k - 16)) & 1:
                xh ^= d[k][None, :]
        x = tbl[g] ^ xh ^ sh[None, :]
        z = _erfinv_cpu(x.to(torch.float32) * (1.0 / 4294967296.0) * 2.0 - 1.0) * math.sqrt(2)
        out[s:s + n] = z
        out[half + s:half + s + n] = -z


def sample_base(sobol: SobolEngine, count, device=None):
    """[2*(count//2), dim] fp32 standard-normal samples, z then -z (antithetic)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dim = sobol.dimension
    half = count // 2
    if half == 0:
        return torch.empty((0, dim), dtype=torch.float32, device=device)
    key = (id(sobol), dim)
    if key not in _cache:
        d = (sobol.sobolstate << 2).t().contiguous()                      # [32, dim] dirs
        sh = ((sobol.shift if sobol.scramble else torch.zeros(dim, dtype=torch.long)) << 2)
        tbl = torch.zeros((65536, dim), dtype=torch.long)
        for k in range(16):
            tbl ^= ((torch.arange(65536) >> k) & 1).to(torch.long)[:, None] * d[k][None, :]
        _cache[key] = (d, sh, tbl)
    d, sh, tbl = _cache[key]
    out = torch.empty((2 * half, dim), dtype=torch.float32, device=device)
    if device.type == "cpu":
        _cpu(out, d, sh, tbl, half, dim)
    else:
        block = 1 << min(8, max(6, (8192 // dim).bit_length() - 1))
        _kernel[(triton.cdiv(half, block),)](
            d.to(device).to(torch.uint32), sh.to(device).to(torch.uint32), out, half,
            DIM=dim, DIMP2=triton.next_power_of_2(dim), BLOCK=block,
            num_warps=min(8, max(2, block // 32)))
    return out
