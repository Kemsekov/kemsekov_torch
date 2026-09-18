import torch
from torch import nn


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
