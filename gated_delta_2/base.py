import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from kemsekov_torch.common_modules import zero_module, StepState


@dataclass
class Delta2State(StepState):
    """
    Recurrent state of a Gated Delta-2 layer.

    ``state`` is the running memory matrix of shape
    ``[batch*heads, QK_dim, V_dim]`` (``None`` until the first
    :meth:`GatedDelta2Base.step` call).
    """
    state: Optional[torch.Tensor] = None

    def detach(self) -> "Delta2State":
        return Delta2State(
            None if self.state is None else self.state.detach()
        )


class GatedDelta2Base(nn.Module):
    def __init__(self, dim, QK_dim, V_dim, heads=1, kv_heads=None, erase_gate_scale=1.0,bidirectional : bool = False):
        super().__init__()
        if kv_heads is None:
            kv_heads = heads
        assert heads % kv_heads == 0, "heads must be divisible by kv_heads"
        self.heads = heads
        self.kv_heads = kv_heads
        self.groups = heads // kv_heads
        self.QK_dim = QK_dim
        self.V_dim = V_dim
        self.dim = dim
        self.pernorm=nn.RMSNorm(dim)
        self.register_buffer(
            "erase_gate_scale", torch.tensor([erase_gate_scale])
        )
        self.decay = nn.Parameter(torch.tensor([0.0]))
        self.projection = nn.Linear(
            dim,
            QK_dim * kv_heads
            + V_dim * kv_heads
            + QK_dim * kv_heads
            + QK_dim * (heads + kv_heads)
            + V_dim * kv_heads,
            bias=False,
        )
        self.decay_bias = nn.Parameter(torch.zeros(QK_dim * kv_heads))
        self.out = nn.Sequential(
            nn.RMSNorm(V_dim * heads),
            nn.SiLU(),
            zero_module(nn.Linear(V_dim * heads, dim)),
        )
        self.bidirectional=bidirectional

    def _move_heads_to_batch(self, x, heads=None):
        heads = self.heads if heads is None else heads
        ndim = x.ndim
        batch, seqlen, mult = x.shape[:3]
        x = x.view(batch, seqlen, heads, -1)
        x = x.transpose(1, 2)
        if ndim == 4:
            return x.reshape(batch * heads, seqlen, -1, 1)
        return x.reshape(batch * heads, seqlen, -1)

    def _expand_kv_heads(self, x):
        """GQA: repeat every key/value head ``self.groups`` times so query head
        ``h`` reads the kv head ``h // self.groups`` (same mapping as
        ``F.scaled_dot_product_attention(..., enable_gqa=True)``)."""
        if self.groups == 1:
            return x
        return x.repeat_interleave(self.groups, dim=0)

    def _move_batch_to_heads(self, x, batch):
        x = x.view(batch, self.heads, -1, x.size(-1))
        x = x.transpose(1, 2)
        return x.reshape(batch, -1, self.heads * x.size(-1))

    @staticmethod
    def _mixing_dtype(xt):
        dt = xt.dtype
        return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt

    def _project(self, xt):
        xt=self.pernorm(xt)
        batch, seqlen, dim = xt.shape
        cdt = self._mixing_dtype(xt)
        erase_h, write_h, decay_h, qk_h, v_h = self.projection(xt).split(
            [
                self.QK_dim * self.kv_heads,
                self.V_dim * self.kv_heads,
                self.QK_dim * self.kv_heads,
                self.QK_dim * (self.heads + self.kv_heads),
                self.V_dim * self.kv_heads,
            ],
            dim=-1,
        )
        decay_h = decay_h + self.decay_bias
        Q, K = qk_h.unsqueeze(-1).split(
            [self.QK_dim * self.heads, self.QK_dim * self.kv_heads], dim=-2
        )
        V = v_h
        Q = self._move_heads_to_batch(Q, self.heads)
        K = self._move_heads_to_batch(K, self.kv_heads)
        V = self._move_heads_to_batch(V, self.kv_heads)
        Q = F.normalize(Q.to(cdt), dim=-2)
        K = F.normalize(K.to(cdt), dim=-2)
        V = V.to(cdt)
        bt = erase_h.sigmoid() * self.erase_gate_scale
        wt = write_h.sigmoid()
        gt = -self.decay.to(cdt).exp() * F.softplus(decay_h)
        bt = self._move_heads_to_batch(bt, self.kv_heads)
        wt = self._move_heads_to_batch(wt, self.kv_heads)
        gt = self._move_heads_to_batch(gt, self.kv_heads)
        alpha = gt.to(cdt).exp().unsqueeze(-1)
        et = bt.to(cdt).unsqueeze(-1) * K
        zt = wt.to(cdt) * V
        K, alpha, et, zt = map(self._expand_kv_heads, (K, alpha, et, zt))
        return batch, seqlen, Q, K, alpha, et, zt

    def _finalize(self, out, batch,xt):
        out = self._move_batch_to_heads(out, batch)
        return self.out(out.to(xt.dtype))+xt

    def init_state(self, batch_size, device=None, dtype=None) -> Delta2State:
        """Create an empty incremental state for :meth:`step`."""
        return Delta2State()

    def step(self, xt, state: Delta2State):
        """
        Incremental forward pass: applies the exact per-token recurrence to a
        chunk of new tokens and updates the running memory state.

        xt: `[B, L, dim]` with new tokens (L is usually 1).

        state: :class:`Delta2State` returned by :meth:`init_state` (or by a
               previous call to this method).

        Returns `(output, state)` with output shaped like ``xt``.
        """
        if self.bidirectional:
            raise NotImplementedError(
                "GatedDelta2.step is undefined for bidirectional mixing"
            )
        with torch.amp.autocast(xt.device.type, enabled=False):
            batch, seqlen, Q, K, alpha, et, zt = self._project(xt)
            assert seqlen>0, "GatedDelta2.step requires at least one token"
            S = state.state
            if S is None:
                S = torch.zeros(
                    Q.shape[0], self.QK_dim, self.V_dim,
                    dtype=alpha.dtype, device=alpha.device
                )
            result = []
            for i in range(seqlen):
                S = alpha[:,i]*S
                rt = (S.transpose(1,2) @ et[:,i]).squeeze(-1)
                S = S + K[:,i]*(zt[:,i]-rt)[:,None]
                result.append((S.transpose(-1,-2) @ Q[:,i])[:,:,0])
            out = torch.stack(result,1)
            state.state = S
        return self._finalize(out, batch, xt), state
