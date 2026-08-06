"""
Static-input CUDA graph capture helper (_CudaGraph).
"""
import torch


class _CudaGraph:
    """
    Captures ``fn(*inputs)`` as a CUDA graph with static input buffers.

    ``inputs`` are pre-allocated static tensors whose contents are copied by
    the caller right before each ``replay()``. ``outputs`` are the tensors
    returned by ``fn`` during capture: each replay recomputes their values
    in place, so they can be read after ``replay()``.

    All kernel-launch and python overhead of the captured computation is
    eliminated on replay, which is the dominant cost for this small model.
    """
    def __init__(self, fn, inputs, n_warmup=3):
        self.inputs = inputs
        self.outputs = None
        # warmup on the current stream so that kernels/caches (and any
        # autograd state such as gradient accumulation nodes) are created on
        # the same stream that the capture will use; a side-stream warmup
        # would create AccumulateGrad nodes on the wrong stream and cause the
        # next capture to reallocate .grad, invalidating earlier graphs
        for _ in range(n_warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.outputs = fn(*inputs)
        # detach keeps the same storage (replays update it in place) but drops
        # the autograd references, so the capture-time autograd graph (and its
        # AccumulateGrad nodes) is freed instead of being kept alive by the
        # cached outputs — otherwise a later capture of a different batch size
        # reallocates .grad and invalidates this graph's captured addresses
        self.outputs = tuple(o.detach() for o in self.outputs)
        self.graph = graph
    @staticmethod
    def capture(fn, inputs):
        """
        Returns a _CudaGraph, or None if the capture fails (in which case the
        caller falls back to the eager path).
        """
        try:
            return _CudaGraph(fn, inputs)
        except Exception:
            return None
    def replay(self):
        self.graph.replay()