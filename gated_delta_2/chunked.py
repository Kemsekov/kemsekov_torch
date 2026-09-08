import torch

from .base import GatedDelta2Base
from .scan import Delta2ScanFn


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
        float32 with automatic float64 fallback when cumulative decay is
        extreme), ``"mixed"`` (float64 only for the decay normalization),
        ``"fp64"`` (everything in float64).
    """

    def __init__(
        self, dim, QK_dim, V_dim, heads=1, erase_gate_scale=1.0, chunk=64,
        scan_mode="auto", prec="fp32",
    ):
        super().__init__(dim, QK_dim, V_dim, heads=heads,
                         erase_gate_scale=erase_gate_scale)
        self.chunk = chunk
        self.scan_mode = scan_mode
        self.prec = prec

    def forward(self, xt):
        batch, seqlen, Q, K, alpha, et, zt = self._project(xt)
        out = Delta2ScanFn.apply(
            alpha, K, et, Q, zt, self.chunk, self.scan_mode, self.prec
        )
        return self._finalize(out, batch,xt)
