"""
kemsekov_torch.fm: flow-matching package.

Modules:
  samplers     - prior sampling + integration samplers
  core         - FlowMatching, LossNormalizer1d, FusedFlowResidual, optim groups
  matching     - approximate point matching (projection + rank alignment)
  cuda_graph   - _CudaGraph static-input capture helper
  model_core   - FlowModel1dCore (network, forward, device/dtype)
  training     - FlowModel1dTrainingMixin (fit / reflow)
  sampling     - FlowModel1dSamplingMixin (transport, sampling, log_prob)
  model        - FlowModel1d (public class)
"""
from kemsekov_torch.fm.samplers import (sample_base, euler, momentum_heun, heun,
                                         rk3, rk2, one_step)
from kemsekov_torch.fm.core import (FlowMatching, LossNormalizer1d, zero_module,
                                    FusedFlowResidual)
from kemsekov_torch.fm.matching import (match_approximate_fast,
                                        match_approximate_random_proj,
                                        match_approximate_sliced)
from kemsekov_torch.fm.cuda_graph import _CudaGraph
from kemsekov_torch.fm.model import FlowModel1d

__all__ = [
    "sample_base", "euler", "momentum_heun", "heun", "rk3", "rk2", "one_step",
    "FlowMatching", "LossNormalizer1d", "zero_module", "FusedFlowResidual",
    "match_approximate_fast", "match_approximate_random_proj",
    "match_approximate_sliced", "_CudaGraph", "FlowModel1d",
]
