import torch

from .base import GatedDelta2Base


def _serial_mix(alpha, et, zt, K, Q):
    alpha = alpha.transpose(0, 1)
    et = et.transpose(0, 1)
    zt = zt.transpose(0, 1)
    K = K.transpose(0, 1)
    Q = Q.transpose(0, 1)
    seqlen = K.shape[0]
    Bp = K.shape[1]
    QK_dim = K.shape[-2]
    V_dim = zt.shape[-1]
    state = torch.zeros(
        Bp, QK_dim, V_dim, dtype=alpha.dtype, device=alpha.device
    )
    result = []
    for i in range(seqlen):
        decayed_state = alpha[i] * state
        rt = decayed_state.transpose(1, 2) @ et[i]
        rt = rt.squeeze(-1)
        diff = zt[i] - rt
        outer = K[i] * diff[:, None]
        state = decayed_state + outer
        out = state.transpose(-1, -2) @ Q[i]
        result.append(out[:, :, 0])
    return torch.stack(result, 1)


class GatedDelta2(GatedDelta2Base):
    """Reference token-by-token (serial loop) implementation.

    Iterates over the sequence with explicit per-token state updates. This is
    the numerically exact reference used to validate the parallel-scan
    implementations and is not meant to be fast on long sequences.
    """

    def forward(self, xt):
        batch, seqlen, Q, K, alpha, et, zt = self._project(xt)
        out = _serial_mix(alpha, et, zt, K, Q)
        return self._finalize(out, batch,xt)
