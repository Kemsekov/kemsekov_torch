import torch

from .base import GatedDelta2Base


def dense_prefix_states(a, k, e, q, z):
    """Inclusive log-depth (Hillis-Steele) scan over *per-token* affine maps.

    Each token is represented by the full dk x dk state-transition map
    ``S_t = (I - k_t e_t^T) diag(a_t) S_{t-1} + k_t z_t^T``, so the scan does
    O(L dk^3 log L) work. Kept for didactic comparison with the chunked WY
    scan in ``scan.py``, which is much cheaper.
    """
    B, L, dk = k.shape
    dv = z.shape[-1]
    dt = a.dtype
    eye = torch.eye(dk, dtype=dt, device=a.device)
    rho = a * e
    A = torch.diag_embed(a) - k.unsqueeze(-1) * rho.unsqueeze(-2)
    b = k.unsqueeze(-1) * z.unsqueeze(-2)
    off = 1
    while off < L:
        As = torch.cat([eye.expand(B, off, dk, dk).clone(), A[:, : L - off]], 1)
        bs = torch.cat([b.new_zeros(B, off, dk, dv), b[:, : L - off]], 1)
        A_new = torch.matmul(A, As)
        b_new = torch.matmul(A, bs) + b
        A = A_new
        b = b_new
        off <<= 1
    return A, b


def dense_scan_forward(a, k, e, q, z):
    _, b = dense_prefix_states(a, k, e, q, z)
    return torch.matmul(b.transpose(-1, -2), q.unsqueeze(-1)).squeeze(-1)


class GatedDelta2ScanDense(GatedDelta2Base):
    """Experimental per-token dense-map scan (no chunking).

    Correctness-wise equivalent to the serial reference, but its cost is
    O(L dk^3 log L) per pass, so it is only useful for short sequences and
    for comparing scan strategies. See ``chunked.py`` for the fast option.
    """

    def forward(self, xt):
        batch, seqlen, Q, K, alpha, et, zt = self._project(xt)
        a = alpha.squeeze(-1).double()
        k = K.squeeze(-1).double()
        e = et.squeeze(-1).double()
        q = Q.squeeze(-1).double()
        z = zt.double()
        out = dense_scan_forward(a, k, e, q, z).float()
        return self._finalize(out, batch)
