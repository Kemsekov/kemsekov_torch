import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDelta2Base(nn.Module):
    def __init__(self, dim, QK_dim, V_dim, heads=1, erase_gate_scale=1.0,bidirectional : bool = False):
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
            nn.Linear(V_dim * heads, dim),
        )
        self.bidirectional=bidirectional

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

    @staticmethod
    def _mixing_dtype(xt):
        dt = xt.dtype
        return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt

    def _project(self, xt):
        batch, seqlen, dim = xt.shape
        cdt = self._mixing_dtype(xt)
        W = torch.cat(
            [
                self.erase_gate.weight,
                self.write_gate.weight,
                self.decay_gate.weight,
                self.QK.weight,
                self.V.weight,
            ],
            dim=0,
        )
        erase_h, write_h, decay_h, qk_h, v_h = F.linear(xt, W).split(
            [
                self.QK_dim * self.heads,
                self.V_dim * self.heads,
                self.QK_dim * self.heads,
                self.QK_dim * 2 * self.heads,
                self.V_dim * self.heads,
            ],
            dim=-1,
        )
        decay_h = decay_h + self.decay_gate.bias
        Q, K = qk_h.unsqueeze(-1).chunk(2, -2)
        V = v_h
        Q, K, V = map(self._move_heads_to_batch, [Q, K, V])
        Q = F.normalize(Q.to(cdt), dim=-2)
        K = F.normalize(K.to(cdt), dim=-2)
        V = V.to(cdt)
        bt = erase_h.sigmoid() * self.erase_gate_scale
        wt = write_h.sigmoid()
        gt = -self.decay.to(cdt).exp() * F.softplus(decay_h)
        bt, wt, gt = map(self._move_heads_to_batch, [bt, wt, gt])
        alpha = gt.to(cdt).exp().unsqueeze(-1)
        et = bt.to(cdt).unsqueeze(-1) * K
        zt = wt.to(cdt) * V
        return batch, seqlen, Q, K, alpha, et, zt

    def _finalize(self, out, batch,xt):
        out = self._move_batch_to_heads(out, batch)
        return self.out(out.to(xt.dtype))+xt
