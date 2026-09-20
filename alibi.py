import torch
from torch import nn
from typing import Optional


class AlibiEmb(nn.Module):
    """
    ALiBi (Attention with Linear Biases, Press et al. 2021) as an additive
    attention mask.

    ``forward`` returns a ``[heads, q_len, kv_len]`` bias

        bias[h, i, j] = -slopes[h] * distance(q_pos_i, k_pos_j)

    where ``distance`` is the clamped lag ``q_pos_i - k_pos_j`` for causal
    attention (positions with ``j > i`` are additionally masked with ``-inf``)
    and the absolute lag ``|q_pos_i - k_pos_j|`` for non-causal attention, so
    bidirectional attention (e.g. over conv activations) gets a symmetric
    distance penalty looking into both directions.
    The slopes default to the geometric schedule ``2 ** (-8h / heads)``,
    ``h = 1..heads``; there is one slope per *query* head, so grouped-query
    attention works unchanged.

    ``q_offset``/``k_offset`` shift the position axes: incremental decoding
    (:meth:`SelfAttention.step`) passes the running query offset so the biases
    stay relative to the cached keys.

    The returned mask is broadcastable to ``[B, heads, q_len, kv_len]`` and can
    be handed directly to :func:`torch.nn.functional.scaled_dot_product_attention`.
    Because the causal mask is part of the bias, ``is_causal`` must be False
    when it is used.
    """

    def __init__(
        self,
        heads: int,
        slopes: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        """
        heads: number of query heads (one ALiBi slope per head).
        slopes: optional custom slopes, shape ``[heads]``. Defaults to
                ``2 ** (-8h / heads)``.
        is_causal: use the causal lag (clamped at zero) and mask future
                   positions with ``-inf``; otherwise use the absolute
                   distance between query and key positions.
        """
        super().__init__()
        if slopes is None:
            slopes = 2.0 ** (-8.0 * torch.arange(1, heads + 1) / heads)
        slopes = torch.as_tensor(slopes, dtype=torch.float32)
        assert slopes.numel() == heads, "AlibiEmb expects one slope per head"
        self.heads = heads
        self.is_causal = is_causal
        # derived from `heads`; keep it out of the state dict
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(self, q_len, kv_len, device=None, dtype=None, q_offset=0, k_offset=0):
        q_pos = torch.arange(q_offset, q_offset + q_len, device=device)
        k_pos = torch.arange(k_offset, k_offset + kv_len, device=device)
        lag = q_pos[:, None] - k_pos[None, :]  # [q_len, kv_len]
        if self.is_causal:
            bias = -self.slopes[:, None, None] * lag.clamp(min=0)
            bias = bias.masked_fill(lag[None] < 0, float("-inf"))
        else:
            bias = -self.slopes[:, None, None] * lag.abs()
        return bias if dtype is None else bias.to(dtype)
