import torch
import torch.nn as nn

class LearnedPosEmb(nn.Module):
    def __init__(self, dim: int, max_length_size: int):
        super().__init__()
        self.learned_position = nn.Parameter(torch.zeros(1, max_length_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        seq_len = x.size(1)
        pos_emb = self.learned_position[:, :seq_len, :]
        return x + pos_emb
