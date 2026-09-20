import torch

from .base import GatedDelta2Base
from .fast import BUDGET, FastScanFn
from .recurrent import RecurrentDelta2Fn, _can_split, _can_use_recurrent
from .recurrent_fast import FastRecurrentFn
from .recurrent_fast import _can_use_recurrent as _can_use_recurrent_fast
from .scan import (
    Delta2ScanFn,
    _needs_fp32_fallback,
    _needs_seq_fallback,
    _scan_fwd,
)
from . import tuning

# a candidate has to agree with the reference chunked scan to this relative
# tolerance before it is allowed to win the autotuning
_AGREE_TOL = 5e-3
# the differentiable scan is only used when every chunk's cumulative log-decay
# stays above this value (its backward has no fp64 recursion)
_AUTOGRAD_LOG_MIN = -20.0


class GatedDelta2Scan(GatedDelta2Base):
    """Chunked parallel-scan implementation of the Gated Delta Rule-2.

    Splits the sequence into chunks of size ``chunk``, normalizes the
    channel-wise decay into the rank-one erase/write factors (WY form), and
    propagates chunk states with a parallel affine scan over the chunks.

    The mixing backend is chosen at runtime (``impl="auto"``): the first time
    a (device, dtype, shape, training) combination is seen the candidates that
    are applicable are benchmarked and verified against the reference chunked
    scan, and the winner is cached on disk (``~/.cache/gd2_tune.json``).
    Set ``GD2_AUTOTUNE=0`` to disable this and use the static heuristic.

    Parameters
    ----------
    dim : int
        Model dimension.
    QK_dim : int
        Key/query dimension per head.
    V_dim : int
        Value dimension per head (also the module output dimension).
    heads : int
        Number of parallel query heads.
    kv_heads : int or None
        Number of key/value heads for grouped-query attention (GQA);
        ``heads`` must be divisible by ``kv_heads``. Defaults to ``heads``
        (no grouping).
    erase_gate_scale : float
        Multiplier applied to the sigmoid erase gate.
    chunk : int
        Chunk size used by the scan (typical values 32-128, default 64).
    scan_mode : str
        ``"scan"`` work-efficient (Blelloch) scan over chunk states,
        ``"seq"`` sequential loop over chunk states, ``"auto"`` chooses
        ``"seq"`` for few chunks and ``"scan"`` otherwise.
    prec : str
        Internal precision of the scan kernel: ``"fp32"``, ``"mixed"`` (same
        as fp32) or ``"fp64"``.
    budget : int or None
        Bytes of cached scan intermediates before the backward pass starts
        recomputing the WY factors (default 256 MB,
        ``GD2_SCAN_BUDGET_MB``).
    fast : bool
        Use the optimized mixing backends (default).  ``False`` restores the
        original backend selection.
    impl : str
        ``"auto"`` (autotuned), ``"triton"`` (fused per-token CUDA kernel),
        ``"split"`` (reference two-level recurrent), ``"scan"`` (optimized
        chunked WY scan), ``"autograd"`` (branch-free differentiable scan;
        the one that ``torch.compile`` can fuse end-to-end).
    """

    def __init__(
        self, dim, QK_dim, V_dim, heads=1, kv_heads=None, erase_gate_scale=1.0,bidirectional=False, chunk=64,
        scan_mode="auto", prec="fp32", budget=None, fast=True, impl="auto",
    ):
        super().__init__(dim, QK_dim, V_dim, heads=heads, kv_heads=kv_heads,
                         erase_gate_scale=erase_gate_scale,bidirectional=bidirectional)
        self.chunk = chunk
        self.scan_mode = scan_mode
        self.prec = prec
        self.budget = BUDGET if budget is None else int(budget)
        self.fast = fast
        self.impl = impl

    # ------------------------------------------------------------------ run
    def _run(self, impl, alpha, K, et, Q, zt, mode):
        if impl == "triton" and _can_use_recurrent_fast(alpha, K, Q, zt):
            return FastRecurrentFn.apply(alpha, K, et, Q, zt)
        if impl == "split" and _can_use_recurrent(alpha, K, Q, zt) and alpha.is_cuda:
            return RecurrentDelta2Fn.apply(alpha, K, et, Q, zt)
        if impl == "autograd":
            return _scan_fwd(alpha, K, et, Q, zt, self.chunk, mode, self.prec)
        return FastScanFn.apply(
            alpha, K, et, Q, zt, self.chunk, mode, self.prec, self.budget
        )

    # ------------------------------------------------------------ heuristics
    def _heuristic(self, alpha, K, et, Q, zt):
        dk = K.shape[-2]
        dv = zt.shape[-1]
        L = zt.shape[-2]
        if _can_use_recurrent_fast(alpha, K, Q, zt):
            if L > 2048 and dk >= 64 and dv >= 64:
                return "scan"
            if _can_split(alpha.shape[0], L, dk, dv):
                return "split"
            return "triton"
        return "scan"

    def _tune_key(self, alpha, K, zt, training):
        dev = (
            torch.cuda.get_device_name(alpha.device)
            if alpha.is_cuda else "cpu"
        )
        return "|".join(
            str(v) for v in (
                "v1", dev, torch.__version__, alpha.dtype, int(training),
                alpha.shape[0], zt.shape[-2], K.shape[-2], zt.shape[-1],
                self.chunk,
            )
        )

    def _make_candidate(self, impl, alpha, K, et, Q, zt, mode):
        def run(training):
            if not training:
                return self._run(impl, alpha, K, et, Q, zt, mode)
            inputs = [
                t.detach().requires_grad_(True) for t in (alpha, K, et, Q, zt)
            ]
            out = self._run(impl, *inputs, mode)
            grads = torch.autograd.grad(
                out.float().square().mean(), inputs, retain_graph=False
            )
            if not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError("non-finite backward: " + impl)
            return out.detach()
        return run

    def _autograd_ok(self, alpha):
        """``_scan_fwd`` normalizes the decay in fp32.  Its backward computes
        products of ``exp(+-cumsum)`` factors, so allowing chunk log-decays
        close to the fp32 limit overflows the *gradient* even when the forward
        is finite (mostly visible in compiled graphs).  Only enable it when
        every chunk stays well inside the fp32 range; otherwise the chunked
        scan, which falls back to fp64 / sequential internally, is used."""
        return not _needs_fp32_fallback(
            alpha.squeeze(-1), self.chunk, log_min=_AUTOGRAD_LOG_MIN
        )

    def _candidates(self, alpha, K, et, Q, zt, mode):
        dk = K.shape[-2]
        dv = zt.shape[-1]
        L = zt.shape[-2]
        cands = []
        if _can_use_recurrent_fast(alpha, K, Q, zt) and L <= 2048 and min(dk, dv) >= 32:
            cands.append(("triton", self._make_candidate("triton", alpha, K, et, Q, zt, mode)))
        cands.append(("scan", self._make_candidate("scan", alpha, K, et, Q, zt, mode)))
        if self._autograd_ok(alpha):
            cands.append(
                ("autograd", self._make_candidate("autograd", alpha, K, et, Q, zt, mode))
            )
        return cands

    # -------------------------------------------------------------- dispatch
    def _mix(self, alpha, K, et, Q, zt):
        if self.scan_mode == "auto":
            nch = (zt.shape[-2] + self.chunk - 1) // self.chunk
            mode = "seq" if nch <= 256 else "scan"
        else:
            mode = self.scan_mode

        impl = self.impl
        if impl == "auto" and self.fast:
            impl = None
        elif impl == "auto":
            # legacy path (fast=False): original backend selection
            if _can_use_recurrent(alpha, K, Q, zt):
                return RecurrentDelta2Fn.apply(alpha, K, et, Q, zt)
            if alpha.device.type != "cuda":
                return Delta2ScanFn.apply(
                    alpha, K, et, Q, zt, self.chunk, mode, self.prec
                )
            if _needs_fp32_fallback(alpha.squeeze(-1), self.chunk):
                return Delta2ScanFn.apply(
                    alpha, K, et, Q, zt, self.chunk, mode, self.prec
                )
            return _scan_fwd(alpha, K, et, Q, zt, self.chunk, mode, self.prec)

        if impl is not None:
            if impl == "autograd" and not self._autograd_ok(alpha):
                impl = "scan"
            return self._run(impl, alpha, K, et, Q, zt, mode)

        # --- impl == "auto" (optimized, autotuned) -------------------------
        if torch.compiler.is_compiling():
            # The fusible differentiable scan (``_scan_fwd``) normalizes the
            # decay in fp32; its compiled backward overflows for chunk decay
            # factors that are still finite in the forward, so the compiler
            # path uses the same safe custom kernels as the reference
            # implementation.  ``impl="autograd"`` remains available for
            # regimes with moderate decay.
            return self._run(
                self._heuristic(alpha, K, et, Q, zt), alpha, K, et, Q, zt, mode
            )

        training = torch.is_grad_enabled()
        if not (tuning.enabled() and zt.shape[-2] >= 32):
            return self._run(
                self._heuristic(alpha, K, et, Q, zt), alpha, K, et, Q, zt, mode
            )

        # cases we never tune: the reference split kernels take minutes to JIT
        # on first use, and the sequential fallback has no alternatives
        if _can_use_recurrent_fast(alpha, K, Q, zt) and _can_split(
            alpha.shape[0], zt.shape[-2], K.shape[-2], zt.shape[-1]
        ):
            return self._run("split", alpha, K, et, Q, zt, mode)
        if alpha.is_cuda and _needs_seq_fallback(alpha.squeeze(-1), self.chunk):
            return self._run("scan", alpha, K, et, Q, zt, mode)

        key = self._tune_key(alpha, K, zt, training)
        impl = tuning.lookup(key)
        if impl is None:
            try:
                sync = (
                    torch.cuda.synchronize if alpha.is_cuda else None
                )
                impl, scores, mems = tuning.tune(
                    self._candidates(alpha, K, et, Q, zt, mode),
                    training, _AGREE_TOL, sync=sync, mem=alpha.is_cuda,
                )
                if impl is None:
                    impl = self._heuristic(alpha, K, et, Q, zt)
                else:
                    tuning.store(key, impl, scores, mems)
            except Exception:
                impl = self._heuristic(alpha, K, et, Q, zt)
        if impl == "autograd" and not self._autograd_ok(alpha):
            impl = "scan"
        return self._run(impl, alpha, K, et, Q, zt, mode)

    def forward(self, xt):
        batch, seqlen, Q, K, alpha, et, zt = self._project(xt)
        out = self._mix(alpha, K, et, Q, zt)
        if self.bidirectional:
            out_flip = self._mix(
                alpha.flip(1), K.flip(1), et.flip(1), Q.flip(1), zt.flip(1)
            ).flip(1)
            out = (out+out_flip)*0.707106 # keep variance
        return self._finalize(out, batch,xt)
