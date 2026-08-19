"""
Backward-compatible re-export shim for the refactored flow-matching package.

The former monolithic flow_matching.py has been split into
kemsekov_torch.fm (see its __init__ docstring). This module re-exports
the same public names so existing imports keep working:

    from kemsekov_torch.flow_matching import FlowModel1d

is now equivalent to:

    from kemsekov_torch.fm import FlowModel1d
"""
from kemsekov_torch.fm import (  # noqa: F401
    FlowModel1d,
    FlowMatching,
    FusedFlowResidual,
    LossNormalizer1d,
    _CudaGraph,
    euler,
    heun,
    match_approximate_fast,
    match_approximate_random_proj,
    match_approximate_sliced,
    momentum_heun,
    one_step,
    rk2,
    rk3,
    sample_base,
    zero_module,
)

__all__ = [
    "FlowModel1d", "FlowMatching", "FusedFlowResidual", "LossNormalizer1d",
    "_CudaGraph", "euler", "heun",
    "match_approximate_fast", "match_approximate_random_proj",
    "match_approximate_sliced", "momentum_heun", "one_step", "rk2", "rk3",
    "sample_base", "zero_module",
]
