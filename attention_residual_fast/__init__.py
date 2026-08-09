"""AttentionResidual2 on fused Triton kernels (fast, leak-free).

Semantics match AttentionResidual2 (production etalon) exactly.
"""
from kemsekov_torch.attention_residual_fast.module import AR2Fast
from kemsekov_torch.attention_residual_fast.kernels import (
    _KVFn, _AttnFn, _OutFn,
)

__all__ = ["AR2Fast"]
