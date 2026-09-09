import torch

from .base import GatedDelta2Base
from .recurrent import RecurrentDelta2Fn, _can_use_recurrent
from .scan import Delta2ScanFn, _needs_seq_fallback, _scan_fwd


class GatedDelta2Scan(GatedDelta2Base):
    """Chunked parallel-scan implementation of the Gated Delta Rule-2.

    Splits the sequence into chunks of size ``chunk``, normalizes the
    channel-wise decay into the rank-one erase/write factors (WY form), and
    propagates chunk states with a parallel affine scan over the chunks.
    See ``scan.py`` for the mixing kernel and its manual backward pass.

    Parameters
    ----------
    dim : int
        Model dimension.
    QK_dim : int
        Key/query dimension per head.
    V_dim : int
        Value dimension per head (also the module output dimension).
    heads : int
        Number of parallel heads.
    erase_gate_scale : float
        Multiplier applied to the sigmoid erase gate.
    chunk : int
        Chunk size used by the scan (typical values 32-128, default 64).
    scan_mode : str
        ``"scan"`` work-efficient (Blelloch) scan over chunk states,
        ``"seq"`` sequential loop over chunk states, ``"auto"`` chooses
        ``"seq"`` for few chunks and ``"scan"`` otherwise.
    prec : str
        Internal precision of the scan kernel: ``"fp32"`` (default, pure
        float32 pipeline with float64 decay normalization), ``"mixed"``
        (same as ``"fp32"``), ``"fp64"`` (everything in float64).
    """

    def __init__(
        self, dim, QK_dim, V_dim, heads=1, erase_gate_scale=1.0,bidirectional=False, chunk=64,
        scan_mode="auto", prec="fp32",
    ):
        super().__init__(dim, QK_dim, V_dim, heads=heads,
                         erase_gate_scale=erase_gate_scale,bidirectional=bidirectional)
        self.chunk = chunk
        self.scan_mode = scan_mode
        self.prec = prec

    def _mix(self, alpha, K, et, Q, zt):
        if _can_use_recurrent(alpha, K, Q, zt):
            return RecurrentDelta2Fn.apply(alpha, K, et, Q, zt)
        if self.scan_mode == "auto":
            nch = (zt.shape[-2] + self.chunk - 1) // self.chunk
            mode = "seq" if nch <= 256 else "scan"
        else:
            mode = self.scan_mode
        if alpha.device.type != "cuda":
            # CPU: the manual-backward chunked Function is ~2x faster than the
            # differentiable _scan_fwd (autograd re-walks a large chunked graph
            # and keeps every intermediate alive), and builds one tiny autograd
            # node, so runtimes are steadier under memory pressure.
            return Delta2ScanFn.apply(
                alpha, K, et, Q, zt, self.chunk, mode, self.prec
            )
        if _needs_seq_fallback(alpha.squeeze(-1), self.chunk):
            return Delta2ScanFn.apply(
                alpha, K, et, Q, zt, self.chunk, mode, self.prec
            )
        return _scan_fwd(alpha, K, et, Q, zt, self.chunk, mode, self.prec)

    def forward(self, xt):
        batch, seqlen, Q, K, alpha, et, zt = self._project(xt)
        out = self._mix(alpha, K, et, Q, zt)
        if self.bidirectional:
            out_flip = self._mix(
                alpha.flip(1), K.flip(1), et.flip(1), Q.flip(1), zt.flip(1)
            ).flip(1)
            out = (out+out_flip)*0.707106 # keep variance
        return self._finalize(out, batch,xt)
