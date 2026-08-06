"""
FlowModel1dCore: network construction, forward pass (with CUDA-graph
forward caching), device/dtype handling and pickle/deepcopy support.
The full public FlowModel1d is assembled from this core plus the
training and sampling mixins (see model.py).
"""
from copy import deepcopy
from typing import Literal, Optional
import torch
import torch.nn as nn
from torch.quasirandom import SobolEngine
from kemsekov_torch.attention_residual import AttentionResidual2
from kemsekov_torch.common_modules import ConstModule, Prod
from kemsekov_torch.fm.core import FusedFlowResidual, FlowMatching, zero_module
from kemsekov_torch.fm.cuda_graph import _CudaGraph


class FlowModel1dCore(nn.Module):
    """
    Fully-connected Flow Matching model for vector-valued data.

    FlowModel1d learns a continuous transport map between a simple Gaussian
    prior distribution and a target data distribution using Flow Matching.
    The model supports unconditional and conditional generation, density
    estimation, latent-space optimization, interpolation, constrained
    generation, and ReFlow distillation into fast one-step or two-step
    generators.

    The architecture combines:

    - Learnable time reparameterization
    - FiLM-style time conditioning
    - Optional FiLM-style external conditioning
    - Residual flow blocks
    - Bidirectional transport between prior and target spaces
    - Adaptive numerical integration
    - ReFlow distillation

    Parameters
    ----------
    in_dim : int
        Dimensionality of input vectors.

    conditional_dim : int | None, optional
        Dimensionality of conditioning vectors.

        When specified, the model becomes conditional and accepts condition
        tensors during training, sampling, interpolation, optimization and
        density estimation.

        Expected condition shape:

        ``[batch_size, conditional_dim]``

        If None, the model operates as an unconditional flow.

    hidden_dim : int, default=64
        Internal feature dimension used throughout the network.

    residual_blocks : int, default=5
        Number of residual flow blocks.

    dropout_p : float, default=0.0
        Dropout probability applied before residual processing.

    device : str, default="cpu"
        Device used for model execution.

    dtype : torch.dtype, default=torch.float32
        Precision of the model weights: torch.float32, torch.bfloat16 or
        torch.float16. All weights are stored and computed in this dtype;
        training and inference both run in the provided precision.

    cuda_graphs : bool, default=True
        Enable CUDA-graph acceleration of the hot paths (forward passes,
        whole integrations, fit/reflow training steps, log_prob). Captured
        graphs replay the exact same kernels, so outputs are identical to
        the eager path; they are invalidated automatically when the model is
        moved or cast.

    default_time_scaler : float, default=10.01
        Initial value of the learnable time-reparameterization coefficient.

        Training times are transformed according to:

        .. math::

            t' = \\frac{\\log((s-1)t + 1)}{\\log(s)}

        where ``s`` is the learnable ``time_scaler`` parameter.

        This biases training toward more difficult regions of transport space.

    Attributes
    ----------
    fm : FlowMatching
        Internal Flow Matching helper responsible for training pair
        generation, integration and ReFlow distillation.

    sobol : torch.quasirandom.SobolEngine
        Sobol sequence generator used for low-discrepancy latent sampling.

    time_scaler : torch.nn.Parameter
        Learnable coefficient controlling time reparameterization.

    conditional_dim : int | None
        Dimension of conditioning vectors.

    in_dim : int
        Input dimensionality.

    hidden_dim : int
        Internal network width.

    default_steps : int
        Default integration step count used by:

        - to_target()
        - to_prior()
        - sample()

        Initially set to 16.

        ReFlow automatically updates this value to the distilled step count.

    fit_history : dict
        Available after training.

        Contains:

        .. code-block:: python

            {
                "loss": [...],
                "r2": [...]
            }

    reflow_history : dict
        Available after ReFlow training.

        Contains:

        .. code-block:: python

            {
                "loss": [...],
                "forward_r2": [...],
                "inverse_r2": [...]
            }

    Notes
    -----
    Conditional training uses classifier-free conditioning.

    During training, condition vectors are randomly replaced with zeros
    according to ``condition_dropout``. This improves robustness and enables
    conditional generation even when conditions are partially missing.

    The same model can be used for:

    - Unconditional generation
    - Conditional generation
    - Density estimation
    - Inverse design
    - Constraint-guided sampling
    - Latent-space interpolation
    - ReFlow acceleration
    """

    def __init__(self, in_dim,conditional_dim = None,hidden_dim=64,residual_blocks=5,dropout_p=0.0,device='cpu',default_time_scaler = 10.01,residual_blocks_impl : Literal['sequential','attention']='sequential',cuda_graphs = True,dtype = torch.float32) -> None:
        super().__init__()
        self.cuda_graphs = bool(cuda_graphs)
        self._fwd_graphs = {}      # (x_shape, t_shape, c_shape) -> _CudaGraph
        self._integ_graphs = {}    # (x_shape, steps, inverse) -> _CudaGraph
        self._train_graphs = {}    # ('fit'|'reflow', batch_size) -> _CudaGraph
        self.fm = FlowMatching()
        self.sobol = SobolEngine(in_dim, scramble=True)
        self.conditional_dim=conditional_dim
        # time scaler for training
        self.time_scaler = torch.nn.Parameter(torch.tensor([float(default_time_scaler)]))
        # this thing will dynamically shift training to harder part of vector-space
        self.fm.time_scaler = lambda x: torch.log((self.time_scaler-1)*x+1)/self.time_scaler.log()
        
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        norm = nn.RMSNorm

        
        self.time_emb = nn.Sequential(
            nn.Linear(1,hidden_dim),
            Prod(nn.Sequential(
                nn.SiLU(),
                # zero_module(nn.Linear(hidden_dim,hidden_dim)),
                nn.Linear(hidden_dim,hidden_dim),
                # nn.RMSNorm(hidden_dim),
                nn.Tanh(),
            )),
            # nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            # nn.Linear(hidden_dim,hidden_dim*2),
            zero_module(nn.Linear(hidden_dim,hidden_dim*2)),
        )
        
        if conditional_dim is not None:
            self.condition_emb = nn.Sequential(
                nn.Linear(conditional_dim,hidden_dim),
                Prod(nn.Sequential(
                    nn.SiLU(),
                    # zero_module(nn.Linear(hidden_dim,hidden_dim)),
                    nn.Linear(hidden_dim,hidden_dim),
                    # nn.RMSNorm(hidden_dim),
                    nn.Tanh(),
                )),
                # nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                # nn.Linear(hidden_dim,hidden_dim*2),
                zero_module(nn.Linear(hidden_dim,hidden_dim*2)),
            )
        else:
            self.condition_emb = ConstModule(0)
            
        self.expand = nn.Linear(in_dim,hidden_dim)
        
        self.dropout = nn.Dropout(dropout_p) if dropout_p>0 else nn.Identity()
        self.norm = norm(hidden_dim)
        allowed = ['sequential','attention']
        assert residual_blocks_impl in allowed,f'residual_blocks_impl must be one of {allowed}'
        if residual_blocks_impl=='sequential':
            self.residual_blocks = nn.Sequential(*[
                        FusedFlowResidual(hidden_dim)
                for i in range(residual_blocks)
            ])
        
        if residual_blocks_impl=='attention':
            self.residual_blocks = AttentionResidual2([
                FusedFlowResidual(hidden_dim)
                for i in range(residual_blocks)
            ],hidden_dim,-1)
        
        self.out_norm = norm(hidden_dim)
        
        self.collapse = nn.Sequential(
            nn.Linear(hidden_dim,in_dim)
        )
        
        self.out_prod = nn.Sequential(
            zero_module(nn.Linear(hidden_dim,in_dim)),
            nn.RMSNorm(in_dim),
        )
        self.default_steps=16
        self.to(device)
        # store model weights in the provided precision
        self.to(dtype)
        self.eval()
    def _param_dtype(self) -> torch.dtype:
        """
        Dtype of the model's floating-point weights.
        """
        for p in self.parameters():
            if p.is_floating_point():
                return p.dtype
        return torch.float32
    def _eager_forward(self,x : torch.Tensor,t : torch.Tensor,condition : Optional[torch.Tensor] = None):
        """
        The forward pass without CUDA-graph acceleration. Bit-identical to
        forward() with cuda_graphs disabled.
        """
        dtype = self._param_dtype()
        x = x.to(self.device, dtype=dtype)
        while t.ndim<x.ndim:
            t = t[:,None]
        t = t.to(dtype=dtype)
        expand = self.expand(x)
        x=x.to(self.device)

        # add time embedding
        time_scale,time_shift = self.time_emb(t).chunk(2,-1)
        
        if self.conditional_dim is not None:
            if condition is None:
                condition=torch.zeros((len(x),self.conditional_dim),device=x.device,dtype=x.dtype)
            else:
                condition=condition.to(self.device, dtype=dtype)
            c_scale,c_shift = self.condition_emb(condition).chunk(2,-1)
            
            # print('before',time_scale.shape,c_scale.shape)

            time_shape = list(time_scale.shape)
            time_shape[0]=max(time_shape[0],c_scale.shape[0])
            c_scale=c_scale.view(time_shape)
            c_shift=c_shift.view(time_shape)
            
            # print('after',time_scale.shape,c_scale.shape)
            
            time_scale=time_scale+c_scale
            time_shift=time_shift+c_shift
        x = expand*(1+time_scale)+time_shift
        
        x = self.dropout(x)
        x = self.norm(x)
        x=self.residual_blocks(x)
        x=self.out_norm(x)
        return self.collapse(x)*self.out_prod(x).sigmoid()
    def _graphs_available(self) -> bool:
        """
        Whether CUDA-graph acceleration is enabled and usable on this model.
        """
        return (self.cuda_graphs and torch.cuda.is_available()
                and str(self.device).startswith('cuda'))
    def _can_graph(self) -> bool:
        """
        Whether the inference forward/integration paths can use CUDA graphs
        (disabled under grad, since gradients need the autograd engine).
        """
        return self._graphs_available() and not torch.is_grad_enabled()
    def _invalidate_graphs(self):
        """
        Drops all captured graphs. Must be called whenever parameter tensors
        are replaced or moved (device/dtype casts, reset_weights), since
        captured graphs reference the old memory addresses.
        """
        self._fwd_graphs.clear()
        self._integ_graphs.clear()
        self._train_graphs.clear()
    def _apply(self, fn, recurse=True):
        # _apply is the common path for to()/half()/bfloat16()/float()/...
        # device and dtype casts; invalidate graphs since they pin memory
        self._invalidate_graphs()
        return super()._apply(fn, recurse)
    def forward(self,x : torch.Tensor,t : torch.Tensor,condition : Optional[torch.Tensor] = None):
        if self._can_graph() and not self.training:
            key = (tuple(x.shape), tuple(t.shape), None if condition is None else tuple(condition.shape))
            cg = self._fwd_graphs.get(key)
            if cg is None:
                xb = torch.empty_like(x)
                tb = torch.empty_like(t)
                cb = None if condition is None else torch.empty_like(condition)
                def fn(xb, tb, cb):
                    return (self._eager_forward(xb, tb, cb),)
                cg = _CudaGraph.capture(fn, [xb, tb, cb])
                if cg is None:
                    self.cuda_graphs = False
                    return self._eager_forward(x, t, condition)
                self._fwd_graphs[key] = cg
            cg.inputs[0].copy_(x)
            cg.inputs[1].copy_(t)
            if cg.inputs[2] is not None:
                cg.inputs[2].copy_(condition)
            cg.replay()
            return cg.outputs[0].clone()
        return self._eager_forward(x, t, condition)
    def to(self, *args, **kwargs):
        """
        Tracks the model device while also supporting dtype casts.
        """
        device = kwargs.get('device', None)
        if device is None and len(args) > 0 and isinstance(args[0], (str, torch.device, int)):
            device = args[0]
        if device is not None:
            self.device = device
        return super().to(*args, **kwargs)
    def __deepcopy__(self, memo):
        """
        Deep-copies the model but not the captured CUDA graphs (a CUDAGraph
        object cannot be copied; the copy simply starts with empty caches).
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ('_fwd_graphs', '_integ_graphs', '_train_graphs'):
                continue
            setattr(result, k, deepcopy(v, memo))
        result._fwd_graphs = {}
        result._integ_graphs = {}
        result._train_graphs = {}
        return result
    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ('_fwd_graphs', '_integ_graphs', '_train_graphs'):
            state.pop(k, None)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fwd_graphs = {}
        self._integ_graphs = {}
        self._train_graphs = {}