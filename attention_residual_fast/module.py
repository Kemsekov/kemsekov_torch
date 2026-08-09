"""AR2Fast: AR2 (etalon) semantics on fused Triton kernels.

Same API and semantics as AttentionResidual2 from kemsekov_torch.attention_residual:
  - per step i: k_i = l2norm(KV(xt_i)); v_i = xt_i
  - attention over keys 0..i (INCLUDING current) with query[i], scale = 1/d
  - softmax over the key axis, weighted sum of values -> out module -> module m
  - final key is NOT l2-normalized (etalon quirk), final attention over all n+1 keys

Differences from the etalon:
  - no in-place buffer hacks -> no memory leak
  - ~2.3x faster forward / ~1.8x faster fwd+bwd (fp32, eager); works under CUDA graphs
  - fused kernels used for d in (16, 32, 64); other dims fall back to exact torch ops
  - bf16: results are bf16-noise-equivalent (fp32 internal accumulation); fp32 matches
    the etalon to ~2e-4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from kemsekov_torch.common_modules import Residual
from kemsekov_torch.attention_residual_fast.kernels import (
    _KVFn, _AttnFn, _OutFn, _kv_torch, _out_torch, _use_fused,
)


def _allocator_is_expandable() -> bool:
    """True when the CUDA caching allocator runs in expandable_segments mode.

    The fused Triton kernels are unreliable under expandable_segments on some
    drivers (intermittent illegal memory access in autograd), so we fall back
    to the pure-torch path in that configuration.
    """
    import os
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in conf.replace(" ", ""):
        return True
    try:
        snap = torch.cuda.memory._snapshot()
        return any(s.get("is_expandable", False) for s in snap.get("segments", []))
    except Exception:
        return False


class AR2Fast(nn.Module):
    """Same API and semantics as AttentionResidual2, with fused kernels."""
    def __init__(self, modules: Iterable[nn.Module], features_dim, features_dimension=1):
        super().__init__()
        self.models = nn.ModuleList(modules)
        self.query = nn.Parameter(torch.randn(len(modules) + 1, features_dim))
        self.KV = nn.Sequential(
            nn.RMSNorm(features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim, bias=False),
        )
        self.out = nn.Sequential(
            nn.RMSNorm(features_dim),
            Residual([nn.SiLU(), nn.Linear(features_dim, features_dim)]),
        )
        self.features_dimension = features_dimension
        self.head_dim = features_dim
        self._use_torch_path = None   # cached path decision

    def forward(self, x: torch.Tensor):
        if self._use_torch_path is None:
            self._use_torch_path = (not x.is_cuda) or _allocator_is_expandable()
        if self._use_torch_path:
            return self._forward_torch(x)
        fd = self.features_dimension
        use_cl = (fd == 1 and x.dim() >= 3)
        if use_cl:
            # images: keep data in channels_last [B,C,H,W]; flat rows are the
            # [B,W,H,C] transpose enumeration (same as etalon's xt), which is
            # a contiguous [P,d] view once the tensor is channels_last.
            xc = x.contiguous(memory_format=torch.channels_last)
            xt = xc.transpose(1, -1)
            x_shape = xt.shape
            P = xt.numel() // self.head_dim
            x_flat = xt.reshape(P, self.head_dim)   # free view
        else:
            xt = x.transpose(fd, -1)
            x_shape = xt.shape
            P = xt.numel() // self.head_dim
            x_flat = xt.reshape(P, self.head_dim)
        scale = 1.0 / self.head_dim
        w_rms_kv = self.KV[0].weight
        W_kv = self.KV[2].weight
        w_out = self.out[0].weight
        b_out = self.out[1].m[1].bias
        W_out = self.out[1].m[1].weight
        alpha = self.out[1].alpha

        n = len(self.models)
        fused = _use_fused(self.head_dim)
        K_buf = torch.empty(n + 1, P, self.head_dim, device=x.device, dtype=x.dtype)
        V_buf = torch.empty_like(K_buf)

        keys, values, kvs = [], [], []
        for i, m in enumerate(self.models):
            if fused:
                k = _KVFn.apply(x_flat, w_rms_kv, W_kv, K_buf, V_buf, i, True)
                kvs.append(k)
                kvs.append(x_flat)
                if i > 0:
                    S = _AttnFn.apply(self.query[i], K_buf, V_buf, i + 1, *kvs)
                else:
                    S = x_flat
                x_next = _OutFn.apply(S, w_out, b_out, W_out, alpha)
            else:
                # non-fused dims: torch ops (autograd-native), attention over stacks.
                # kvs must be interleaved [k0, v0, k1, v1, ...] to match _AttnFn.backward.
                k = _kv_torch(x_flat, w_rms_kv, W_kv, True)
                keys.append(k)
                values.append(x_flat)
                if i > 0:
                    kvs = [t for pair in zip(keys, values) for t in pair]
                    S = _AttnFn.apply(self.query[i], torch.stack(keys), torch.stack(values),
                                      i + 1, *kvs)
                else:
                    S = x_flat
                x_next = _out_torch(S, w_out, b_out, W_out, alpha)
            x_next = x_next.view(x_shape).transpose(-1, fd)
            x = m(x_next)
            x_flat = x.transpose(fd, -1).reshape(P, self.head_dim)

        if fused:
            k = _KVFn.apply(x_flat, w_rms_kv, W_kv, K_buf, V_buf, n, False)
            kvs.append(k)
            kvs.append(x_flat)
            S = _AttnFn.apply(self.query[-1], K_buf, V_buf, n + 1, *kvs)
            out = _OutFn.apply(S, w_out, b_out, W_out, alpha)
        else:
            k = _kv_torch(x_flat, w_rms_kv, W_kv, False)
            keys.append(k)
            values.append(x_flat)
            kvs = [t for pair in zip(keys, values) for t in pair]
            S = _AttnFn.apply(self.query[-1], torch.stack(keys), torch.stack(values),
                              n + 1, *kvs)
            out = _out_torch(S, w_out, b_out, W_out, alpha)
        return out.view(x_shape).transpose(-1, fd)

    def _forward_torch(self, x: torch.Tensor):
        """Exact torch fallback (CPU-safe, no Triton). Mirrors the AR2 etalon:
        k_i = l2norm(KV(xt_i)), attention over keys 0..i (incl. current) with
        scale 1/d, softmax over keys, weighted sum of values -> out -> module m;
        final key is NOT l2-normalized (etalon quirk)."""
        fd = self.features_dimension
        xt = x.transpose(fd, -1)
        keys, values = [], []
        for i, m in enumerate(self.models):
            k = F.normalize(self.KV(xt), 2.0, -1)
            v = xt
            q = self.query[i]
            keys.append(k)
            values.append(v)
            if i > 0:
                scores = (torch.stack(keys) * q.unsqueeze(0)).mean(-1, keepdim=True)
                x_next = self.out((torch.stack(values, 0) * scores.softmax(0)).sum(0))
                x_next = x_next.transpose(-1, fd)
            else:
                x_next = self.out(v).transpose(-1, fd)
            x = m(x_next)
            xt = x.transpose(fd, -1)
        keys.append(self.KV(xt))
        values.append(xt)
        scores = (torch.stack(keys) * self.query[-1].unsqueeze(0)).mean(-1, keepdim=True)
        out = self.out((torch.stack(values, 0) * scores.softmax(0)).sum(0))
        return out.transpose(-1, fd)
