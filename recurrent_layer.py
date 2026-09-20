import torch
from torch import nn

from kemsekov_torch.common_modules import init_module_state, step_module


def zero_module(module):
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
    return module


class RecurrentLayer(nn.Module):
    def __init__(self, module, in_dim, max_recurrence=8, mlp_factor=1):
        super().__init__()
        if not 1 <= max_recurrence < 10:
            raise ValueError("max_recurrence must be between 1 and 9")
        self.module = module
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim * mlp_factor),
            nn.RMSNorm(in_dim * mlp_factor),
            nn.SiLU(),
            zero_module(nn.Linear(in_dim * mlp_factor, in_dim * 2)),
        )
        self.max_recurrence = max_recurrence
        self.res_w = nn.Sequential(
                    nn.RMSNorm(in_dim),
                    nn.SiLU(),
                    zero_module(nn.Linear(in_dim, 1)),
                )
        self.register_buffer("_train_steps", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, x):
        x_in = x
        x0 = self.module(x)
        x = x0
        n = self.max_recurrence - 1
        depth = None
        if self.training:
            self._train_steps += 1
            s = self._train_steps
            full = x0.new_full((), self.max_recurrence)
            cand = ((s - 1) % self.max_recurrence) + 1
            depth = full.where(s > 100, cand)
        for i in range(n):
            update_gate, pass_gate = self.gate(x).sigmoid().chunk(2, -1)
            if depth is not None:
                update_gate = update_gate * (i < depth).to(update_gate.dtype)
            x = update_gate * self.module(x * pass_gate) + (1 - update_gate) * x
        return x + x0 + self.res_w(x0) * x_in

    def init_state(self, batch_size, device=None, dtype=None):
        """
        One inner state per `self.module` application: the module is called
        `max_recurrence` times per token (once for `x0` and once per
        refinement step) and each application attends to its own history, so
        each needs its own KV/recurrent state.
        """
        return [
            init_module_state(self.module,batch_size,device=device,dtype=dtype)
            for _ in range(self.max_recurrence)
        ]

    def step(self, x, states):
        """
        Incremental equivalent of :meth:`forward` for one chunk of new
        positions: replays the recurrence with per-application states, exactly
        mirroring the full-sequence computation for the new positions.
        """
        x_in = x
        x0, s0 = step_module(self.module,x,states[0])
        x = x0
        new_states = [s0]
        for i in range(self.max_recurrence - 1):
            update_gate, pass_gate = self.gate(x).sigmoid().chunk(2, -1)
            out, s = step_module(self.module,x * pass_gate,states[i + 1])
            x = update_gate * out + (1 - update_gate) * x
            new_states.append(s)
        return x + x0 + self.res_w(x0) * x_in, new_states
