import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDelta2Base(nn.Module):
    def __init__(self, dim, QK_dim, V_dim, heads=1, erase_gate_scale=1.0):
        super().__init__()
        self.heads = heads
        self.QK_dim = QK_dim
        self.V_dim = V_dim
        self.dim = dim
        self.erase_gate = nn.Linear(dim, QK_dim * heads, bias=False)
        self.register_buffer(
            "erase_gate_scale", torch.tensor([erase_gate_scale])
        )
        self.write_gate = nn.Linear(dim, V_dim * heads, bias=False)
        self.decay = nn.Parameter(torch.tensor([0.0]))
        self.decay_gate = nn.Linear(dim, QK_dim * heads)
        self.QK = nn.Linear(dim, QK_dim * 2 * heads, bias=False)
        self.V = nn.Linear(dim, V_dim * heads, bias=False)
        self.out = nn.Sequential(
            nn.RMSNorm(V_dim * heads),
            nn.SiLU(),
            nn.Linear(V_dim * heads, V_dim),
        )
        self.residual = nn.Identity() if V_dim==dim else nn.Linear(dim,V_dim)

    def _move_heads_to_batch(self, x):
        ndim = x.ndim
        batch, seqlen, mult = x.shape[:3]
        x = x.view(batch, seqlen, self.heads, -1)
        x = x.transpose(1, 2)
        if ndim == 4:
            return x.reshape(batch * self.heads, seqlen, -1, 1)
        return x.reshape(batch * self.heads, seqlen, -1)

    def _move_batch_to_heads(self, x, batch):
        x = x.view(batch, self.heads, -1, x.size(-1))
        x = x.transpose(1, 2)
        return x.reshape(batch, -1, self.heads * x.size(-1))

    def _project(self, xt):
        batch, seqlen, dim = xt.shape
        Q, K = self.QK(xt).unsqueeze(-1).chunk(2, -2)
        V = self.V(xt)
        Q, K, V = map(self._move_heads_to_batch, [Q, K, V])
        Q = F.normalize(Q, dim=-2)
        K = F.normalize(K, dim=-2)
        bt = self.erase_gate(xt).sigmoid() * self.erase_gate_scale
        wt = self.write_gate(xt).sigmoid()
        gt = -self.decay.exp() * F.softplus(self.decay_gate(xt))
        bt, wt, gt = map(self._move_heads_to_batch, [bt, wt, gt])
        alpha = gt.exp().unsqueeze(-1)
        et = bt.unsqueeze(-1) * K
        zt = wt * V
        return batch, seqlen, Q, K, alpha, et, zt

    def _finalize(self, out, batch,xt):
        out = self._move_batch_to_heads(out, batch)
        return self.out(out)+self.residual(xt)
