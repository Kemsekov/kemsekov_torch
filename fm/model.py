"""
FlowModel1d: the public flow-matching model, assembled from
FlowModel1dCore (network + forward) and the training/sampling mixins.
"""
import torch.nn as nn
from kemsekov_torch.fm.model_core import FlowModel1dCore
from kemsekov_torch.fm.training import FlowModel1dTrainingMixin
from kemsekov_torch.fm.sampling import FlowModel1dSamplingMixin


class FlowModel1d(FlowModel1dCore, FlowModel1dTrainingMixin,
                  FlowModel1dSamplingMixin):
    """Fully-connected Flow Matching model (see FlowModel1dCore for
    the architecture docstring)."""
    pass
