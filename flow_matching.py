from copy import deepcopy
import gc
import math
import os
from typing import Callable, Literal, Optional
from kemsekov_torch.common_modules import Prod, Residual
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.common_modules import mmd_rbf
from kemsekov_torch.log_prop_approx import log_prob_inverse, log_prob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.func import vmap, jacrev
from torch.quasirandom import SobolEngine

def sample_base(sobol : SobolEngine,count,device):
    half=count//2
    # efficient uniform-like space coverage sobol standard normal distribution sampler
    u = sobol.draw(half).to(device)           # [count, latent_dim] in [0, 1]
    z = torch.erfinv(2 * u - 1) * math.sqrt(2)      # Transform to N(0, 1)
    # reduce variance
    return torch.concat([z,-z],0)[:count]

def euler(model, x0, steps, churn_scale=0.0, inverse=False,return_intermediates = False, time_transform : nn.Module = nn.Identity(),no_grad_model=False):
    """
    Samples from a flow-matching model with Euler integration.

    Args:
        model: Callable vθ(x, t) predicting vector field/motion.
        x0: Starting point (image or noise tensor).
        steps: Number of Euler steps.
        churn_scale: Amount of noise added for stability each step.
        inverse (bool): If False, integrate forward from x0 to x1 (image → noise).
                        If True, reverse for noise → image.
        return_intermediates: to return intermediates values of xt or not.
    Returns:
        Tuple[Tensor,List[Tensor]]:
        1. xt - Final sample tensor.
        2. intermediates - Intermediate xt values if return_intermediates is True
    """
    device = list(model.parameters())[0].device
    if inverse:
        ts = torch.linspace(1, 0, steps, device=device)
    else:
        ts = torch.linspace(0, 1, steps, device=device)
    # ts = time_transform(ts[:,None])[:,0]
    
    if len(ts)>1:
        dt = ts[1]-ts[0]
    else:
        dt = min(1-ts[0],ts[0])
        
    x0 = x0.to(device)
    xt = x0
    
    intermediates = []
    
    def no_grad_model_pred(xt,t):
        with torch.no_grad():
            return model(xt,t)
    
    pred_m = no_grad_model_pred if no_grad_model else model
    
    for i in range(0,steps):
        t = ts[i]
        
        # optional churn noise
        if churn_scale>0:
            noise = xt.std() * torch.randn_like(xt) + xt.mean()
            xt = churn_scale * noise + (1 - churn_scale) * xt
        t_expand = t.expand(x0.shape[0])
        
        pred = pred_m(xt, t_expand)
        
        # forward or reverse Euler update
        xt = xt + dt * pred
        if return_intermediates:
            intermediates.append(xt)
    
    if return_intermediates:
        return xt, intermediates
    return xt

def heun(model, x0, steps, churn_scale=0.0, inverse=False, return_intermediates=False, time_transform : nn.Module = nn.Identity(),no_grad_model = False):
    device = x0.device
    if inverse:
        ts = torch.linspace(1, 0, steps+1, device=device)  # steps intervals = steps+1 points
    else:
        ts = torch.linspace(0, 1, steps+1, device=device)
        ts = time_transform(ts[:,None])[:,0]
    dt = ts[1:]-ts[:-1]
    x0 = x0.to(device)
    xt = x0
    intermediates = []
    
    # Store previous derivative for multi-step method
    prev_pred = None
    
    def no_grad_model_pred(xt,t):
        with torch.no_grad():
            return model(xt,t)
    
    pred = no_grad_model_pred if no_grad_model else model
    
    for i in range(steps):
        t_current = ts[i]
        t_next = ts[i+1]
        
        # Optional churn noise
        if churn_scale > 0:
            noise = xt.std() * torch.randn_like(xt) + xt.mean()
            xt = churn_scale * noise + (1 - churn_scale) * xt
        
        t_expand_current = t_current.expand(x0.shape[0])
        
        if i == 0:
            # First step: full Heun evaluation (2 evaluations)
            # Current derivative
            pred_current = pred(xt, t_expand_current)
            
            # Predictor step (Euler)
            x_pred = xt + dt[i] * pred_current
            
            # Evaluate at predicted point
            t_expand_next = t_next.expand(x0.shape[0])
            pred_next = pred(x_pred, t_expand_next)
            
            # Corrector step (Heun's method)
            xt = xt + dt[i] * 0.5 * (pred_current + pred_next)
            
            # Store the predictor derivative for next step
            prev_pred = pred_next
        else:
            # Subsequent steps: reuse previous derivative (1 evaluation per step)
            # Predictor using previous derivative (Euler step)
            x_pred = xt + dt[i] * prev_pred
            
            # Evaluate at predicted point (ONLY ONE EVAL PER STEP)
            t_expand_next = t_next.expand(x0.shape[0])
            pred_next = pred(x_pred, t_expand_next)
            
            # Corrector step using stored previous derivative
            xt = xt + dt[i] * 0.5 * (prev_pred + pred_next)
            
            # Update stored derivative for next step
            prev_pred = pred_next
        
        if return_intermediates:
            intermediates.append(xt.clone())
    
    if return_intermediates:
        return xt, intermediates
    return xt

def rk3(model, x0, churn_scale=0.0, inverse=False, return_intermediates=False, left = 0.0, right = 1.0):
    device = next(model.parameters()).device
    x0 = x0.to(device)
    xt = x0.clone()
    
    # === THEORETICAL DEFAULTS: Classical RK3 Butcher tableau ===
    # Forward: t ∈ [0, 1], Reverse: t ∈ [1, 0] (proper time mapping)
    # but i expect these values to be altered for flow matching model towards
    # the center by some amount
    if inverse:
        t_start =   right
        t_end =     left
    else:
        t_start =   left
        t_end =     right
    
    
    dt = t_end - t_start  # = -1.0 for reverse, +1.0 for forward
    
    intermediates = []
    
    if churn_scale > 0:
        noise = xt.std() * torch.randn_like(xt) + xt.mean()
        xt = churn_scale * noise + (1 - churn_scale) * xt
    
    # === First evaluation (k1 at start) - weight = 1/6 ===
    t_expand_start = torch.tensor([t_start], device=device).expand(x0.shape[0])
    k1 = model(xt, t_expand_start)
    
    # === Second evaluation (k2 at midpoint) - weight = 4/6 ===
    t_mid = t_start + dt/2  # = 0.5 for both directions
    x_mid = xt + (dt/2) * k1
    t_expand_mid = torch.tensor([t_mid], device=device).expand(x0.shape[0])
    k2 = model(x_mid, t_expand_mid)
    
    # === Third evaluation (k3 at endpoint) - weight = 1/6 ===
    x_end_predictor = xt + dt * k2  # Classical RK3 uses k2 here, not the complex formula
    t_expand_end = torch.tensor([t_end], device=device).expand(x0.shape[0])
    k3 = model(x_end_predictor, t_expand_end)
    
    # === Classical RK3 update - theoretically optimal weights ===
    xt_next = xt + dt * (k1 + 4*k2 + k3) / 6.0
    
    if return_intermediates:
        intermediates.extend([x_mid, x_end_predictor, xt_next])
    
    return (xt_next, intermediates) if return_intermediates else xt_next

def rk2(model, x0, weights):
    device = next(model.parameters()).device
    x0 = x0.to(device)
    
    t0,t_mid,h1,h2,half,w1,w2,w3 = weights
    
    k1 = model(x0, t0.unsqueeze(0))
    k2 = model(x0+half*k1, t_mid.unsqueeze(0))
    x1 = h1*k1+h2*k2+w1*x0#+k1.pow(2)*k1.sign()*w2+k2.pow(2)*k2.sign()*w3
    return x1

def one_step(model,x0 : torch.Tensor,weights):
    """One-step integration"""
    t=weights[0].unsqueeze(0)
    
    x0_pow2=x0.pow(2)*x0.sign()
    x0_term_add=x0_pow2*weights[3]
    # x0_term_arg=x0_pow2*weights[4]
    pred = model(x0,t)
    
    # pred_2term = pred.pow(2)*pred.sign()*weights[4]
    
    return weights[1]*pred+weights[2]*x0+x0_term_add

class FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_scaler = lambda x:x
        self.reset_weights()
        # weights for one-step integration
    
    def freeze(self):
        self.one_weights.requires_grad_(False)
        self.one_weights_inv.requires_grad_(False)
        self.rk2_weights.requires_grad_(False)
        self.rk2_weights_inv.requires_grad_(False)
    
    def unfreeze(self):
        self.one_weights.requires_grad_(True)
        self.one_weights_inv.requires_grad_(True)
        self.rk2_weights.requires_grad_(True)
        self.rk2_weights_inv.requires_grad_(True)
    
        
    def reset_weights(self):
        if hasattr(self,'one_weights'):
            device = self.one_weights.device
        else:
            device=None
        with torch.no_grad():
            start_time = self.time_scaler(0.5)
            self.one_weights     = torch.nn.Parameter(torch.tensor([start_time,  0.5, 1,0,0],device=device))
            self.one_weights_inv = torch.nn.Parameter(torch.tensor([1-start_time,-0.5,1,0,0],device=device))
            self.rk2_weights     = torch.nn.Parameter(torch.tensor([start_time,   1.0,  0.5, 0.5, 1.0, 1.0, 0.0, 0.0],device=device))
            self.rk2_weights_inv = torch.nn.Parameter(torch.tensor([1-start_time, 0.0, -0.5, -0.5, -1.0, 1.0, 0.0, 0.0],device=device))
    
    def flow_matching_pair(self,model,input_domain,target_domain, time = None):
        """
        Generates direction pairs for flow matching model training
        
        Parameters:
            model: 
                model(xt,t) -> direction prediction. 
                
                Takes linear combination of `input_domain` and `target_domain`
                
                `xt=(1-t)*input_domain+t*target_domain`
                
                `time` is vector of size `[BATCH]` in range `[0;1]`

            input_domain: 
                simple domain (standard normal noise)
                
            target_domain: 
                complex domain (images,time series, etc)
            
            time:
                time to sample inputs. If None, the random time in range [0;1] is generated
        
        Returns:
            Tuple[Tensor,Tensor]:
            1. Predicted direction
            2. Ground truth direction
            3. Time
        """
        # generate time in range [0;1]
        if time is None:
            time = torch.rand(input_domain.shape[0],device=input_domain.device)
        time = self.time_scaler(time)
            
        time_expand = time[:,*([None]*(target_domain.dim()-1))]
        xt = (1-time_expand)*input_domain+time_expand*target_domain
        
        pred_direction = model(xt,time)
        
        #original
        target = (target_domain-input_domain)
        
        return pred_direction,target, time_expand

    def contrastive_flow_matching_pair(self, model, input_domain, target_domain, time=None):
        """
        Generates flow matching training pairs along with contrastive pairs for 
        Contrastive Flow Matching (CFM).

        This extends standard flow matching by returning an additional "negative"
        direction vector. That vector can be used in a contrastive loss as 
        proposed in the paper *Contrastive Flow Matching (ΔFM)*.

        The method constructs interpolated states between `input_domain` and 
        `target_domain` at sampled times, computes the model’s prediction, 
        and provides both ground truth and negative direction vectors.

        Parameters
        ----------
        model : Callable
            Function or neural network of signature `model(x_t, t) -> direction_pred`.
            It predicts the flow direction given interpolated samples `x_t` and time `t`.

        input_domain : torch.Tensor
            Tensor representing the "simple" domain (e.g., standard Gaussian noise).
            Shape: `[B, ...]`

        target_domain : torch.Tensor
            Tensor representing the "complex" domain (e.g., images, time series).
            Shape: `[B, ...]`, same as `input_domain`.

        time : torch.Tensor, optional
            Tensor of shape `[B]` with values in `[0, 1]` representing interpolation 
            times. If None, random times are sampled uniformly.

        Returns
        -------
        pred_direction : torch.Tensor
            Model-predicted flow direction at interpolated state `x_t`.

        target_direction : torch.Tensor
            Ground truth flow direction (from `input_domain` to `target_domain`), 
            scaled by the time-dependent factor.

        contrastive_direction : torch.Tensor
            Ground truth direction vector sampled from a *different* element in the batch.
            This serves as the negative direction for contrastive loss.

        time_expand : torch.Tensor
            Time tensor broadcasted to match input dimensions for interpolation.

        Notes
        -----
        - Contrastive directions are generated by randomly shuffling the batch and 
        recomputing ground truth flow directions.
        - Loss should be computed externally, for example:

        >>> lambda_cf=0.05
        >>> mse_loss = F.mse_loss(pred_direction, target_direction)
        >>> contrastive_loss = F.mse_loss(pred_direction, contrastive_direction)
        >>> loss = mse_loss - lambda_cf * contrastive_loss
        """
        if time is None:
            time = torch.rand(input_domain.shape[0], device=input_domain.device)
        
        time = self.time_scaler(time)
            
        bsz = input_domain.shape[0]
        time_expand = time[:, *([None] * (target_domain.dim() - 1))]
        xt = torch.lerp(input_domain,target_domain,time_expand)
        # xt = (1 - time_expand) * input_domain + time_expand * target_domain
        pred_direction = model(xt, time)

        target = (target_domain - input_domain)

        # Prepare negative samples by shuffling the batch
        idx = torch.randperm(bsz, device=input_domain.device)
        input_neg = input_domain[idx]
        target_neg = target_domain[idx]

        target_neg_vec = (target_neg - input_neg)

        return pred_direction, target, target_neg_vec, time_expand
    def integrate(self,model, x0, steps, churn_scale=0.0, inverse=False, return_intermediates=False,no_grad_model =False):
        """
        Integrates the flow matching model using different numerical methods based on the number of steps.

        This method selects an appropriate numerical integration technique depending on the number of steps:
        - 1 step: Uses Euler method
        - 2 steps: Uses Runge-Kutta 2nd order (RK2) method
        - 3 steps: Uses Runge-Kutta 3rd order (RK3) method
        - More than 3 steps: Uses Heun's method (modified trapezoidal rule)

        Args:
            model: Callable vθ(x, t) predicting vector field/motion.
            x0: Starting point (image or noise tensor).
            steps: Number of integration steps. Determines which numerical method to use.
            churn_scale: Amount of noise added for stability each step.
            inverse (bool): If False, integrate forward from x0 to x1 (image → noise).
                            If True, reverse for noise → image.
            return_intermediates: Whether to return intermediate values of xt.
            no_grad_model: Whether to compute model predictions without gradient tracking.

        Returns:
            Tuple[Tensor,List[Tensor]] or Tensor:
            - Final sample tensor if return_intermediates is False
            - Tuple of (final tensor, list of intermediate tensors) if return_intermediates is True
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0) 
        
        if isinstance(steps,torch.Tensor):
            steps=steps.int().item()
        match steps:
            case 1: return one_step(model,x0,self.one_weights_inv if inverse else self.one_weights)
            case 2: return rk2(model,x0,self.rk2_weights_inv if inverse else self.rk2_weights)
            case 3: return rk3(model,x0,churn_scale,inverse,return_intermediates)
            case _: return heun(model,x0,steps-1,churn_scale,inverse,return_intermediates,time_transform=self.time_scaler,no_grad_model=no_grad_model)

class LossNormalizer1d(nn.Module):
    """
    A neural network module that learns to normalize loss values based on input data and time.

    This module is used in flow matching models to predict appropriate weights for loss normalization,
    helping to stabilize training by adapting the loss function based on the current state and time.

    Attributes:
        expand (nn.Linear): Linear layer to expand input dimension to hidden dimension
        time (nn.Linear): Linear layer to process time embeddings
        net (nn.Sequential): Sequential network processing the combined input and time features
    """
    def __init__(self, in_dim,hidden_dim=32) -> None:
        super().__init__()
        self.expand = nn.Linear(in_dim,hidden_dim)
        self.time = nn.Linear(1,hidden_dim)
        norm = nn.RMSNorm
        self.net = nn.Sequential(
            norm(hidden_dim),
            Residual([
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ]),
            Prod(
                nn.Sequential(
                    nn.Linear(hidden_dim,hidden_dim),
                    norm(hidden_dim),
                    nn.Tanh(),
                )
            ),
            norm(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
            # nn.Softplus()
        )
    def forward(self,x : torch.Tensor,t : torch.Tensor):
        """
        Forward pass of the loss normalizer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_dim]
            t (torch.Tensor): Time tensor of shape [batch_size, 1] or [batch_size]

        Returns:
            torch.Tensor: Normalized loss weights of shape [batch_size, in_dim]
        """
        time = self.time(t)
        x = self.expand(x)
        while time.ndim<x.ndim:
            time = time[:,None]
        return self.net(x+time)

def zero_module(module):
    """
    Zero out the parameters of a module and return it to implement Re-Zero
    """
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
    return module
class FusedFlowResidual(nn.Module):
    def __init__(self,hidden_dim) -> None:
        super().__init__()
        self.prod = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim,bias=False),
            # nn.Tanh()
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(hidden_dim,hidden_dim,bias=False))
        )
        
    def forward(self,x):
        prod = x*self.prod(x)
        return self.out(prod)+x
    
from kemsekov_torch.attention_residual import *
from kemsekov_torch.common_modules import AddConst, ConstModule

def get_fm_optim_groups(model, extra_model=None, weight_decay=1e-2):
    decay_params = []
    no_decay_params = []
    
    def process_model(m):
        for mn, module in m.named_modules():
            # recurse=False ensures we only process parameters directly belonging to this module
            for pn, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                
                # Rule 1: Biases should NEVER be decayed
                if pn.endswith('bias'):
                    no_decay_params.append(p)
                # Rule 2: Normalization layer weights should NEVER be decayed
                elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d, nn.GroupNorm)):
                    no_decay_params.append(p)
                # Rule 3: Protect your custom time_scaler from being shrunk to 0
                elif 'scaler' in pn.lower():
                    no_decay_params.append(p)
                # Rule 4: Everything else (Linear weights, etc.) gets weight decay
                else:
                    decay_params.append(p)

    process_model(model)
    if extra_model is not None:
        process_model(extra_model)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

class FlowModel1d(nn.Module):
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

    def __init__(self, in_dim,conditional_dim = None,hidden_dim=64,residual_blocks=5,dropout_p=0.0,device='cpu',default_time_scaler = 10.01) -> None:
        super().__init__()
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

        self.residual_blocks = nn.Sequential(*[
            FusedFlowResidual(hidden_dim)
            for i in range(residual_blocks)
        ])
        self.out_norm = norm(hidden_dim)
        
        self.collapse = nn.Sequential(
            nn.Linear(hidden_dim,in_dim)
        )
        
        self.out_prod = nn.Sequential(
            zero_module(nn.Linear(hidden_dim,in_dim)),
        )
        self.default_steps=16
        self.to(device)
        self.eval()
    def forward(self,x : torch.Tensor,t : torch.Tensor,condition : Optional[torch.Tensor] = None):
        while t.ndim<x.ndim:
            t = t[:,None]
        expand = self.expand(x)
        x=x.to(self.device)

        # add time embedding
        time_scale,time_shift = self.time_emb(t).chunk(2,-1)
        
        if self.conditional_dim is not None:
            if condition is None:
                condition=torch.zeros((len(x),self.conditional_dim),device=x.device,dtype=x.dtype)
            else:
                condition=condition.to(self.device)
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
        return self.collapse(x)+self.out_prod(x)
    def to(self,device):
        self.device=device
        return super().to(device)
    def fit(
        self,
        data: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_dropout: float = 0.5,
        epochs: int = 64,
        batch_size: int = 256,
        contrastive_loss_weight=1.0,
        lr: float = 0.02,
        distribution_matching=0.0,
        debug: bool = False,
        scheduler = True,
    ):
        """
        Train the Flow Matching model.

        Training uses Contrastive Flow Matching (CFM) together with optional
        distribution matching and classifier-free conditioning.

        During training, random interpolation points are sampled between
        Gaussian prior samples and target data samples. The model learns to
        predict the transport direction connecting both domains.

        When the model is conditional, condition vectors may be randomly
        replaced with zeros according to ``condition_dropout``. This implements
        classifier-free conditioning and improves generalization.
        
        **Model is fitted enough when r2 metric is above 0.36 or so.**

        Parameters
        ----------
        data : torch.Tensor | numpy.ndarray
            Training dataset.

            Expected shape:

            ``[num_samples, in_dim]``

        condition : torch.Tensor | numpy.ndarray | None, optional
            Conditioning vectors.

            Required when training a conditional model.

            Expected shape:

            ``[num_samples, conditional_dim]``

            If None and the model is unconditional, no conditioning is used.

        condition_dropout : float, default=0.5
            Probability of replacing condition vectors with zeros during
            training.

            This implements classifier-free conditioning.

            Values between 0.1 and 0.5 typically work well.

        epochs : int, default=64
            Number of training epochs.

        batch_size : int, default=256
            Mini-batch size.

        contrastive_loss_weight : float, default=1.0
            Weight applied to the contrastive flow matching objective.

            Larger values increase separation from negative transport
            directions.

        lr : float, default=0.02
            Learning rate.

        distribution_matching : float, default=0.0
            Enables adaptive loss reweighting that encourages better matching
            of the target distribution.

            Recommended values:

            - 0.0: disabled
            - 0.05 - 0.25: mild correction
            - >0.5: aggressive distribution matching

        debug : bool, default=False
            Print training statistics.

        scheduler : bool, default=True
            Enable cosine annealing learning-rate scheduling.

        Returns
        -------
        None

        Notes
        -----
        Training history is stored in:

        ``self.fit_history["loss"]``

        ``self.fit_history["r2"]``

        The best model checkpoint is automatically restored at the end of
        training according to validation R² measured on transport direction
        prediction.
        """
        self.unfreeze()
        # these are optimal for cpu-training
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
        except: pass
        gc.disable()
        
        data, condition = self.__prepare_data(data, condition)
        
        model = self
        device = model.device
        batch_size = min(batch_size,data.shape[0])
        
        trainable_weights = list(model.parameters())
        loss_normalizer=None
        if distribution_matching>0:
            loss_normalizer = LossNormalizer1d(model.in_dim,model.hidden_dim).to(device)
            # loss_normalizer = torch.jit.trace(loss_normalizer,(torch.randn((1,self.in_dim),device=device),torch.randn((1,1),device=device)))
            trainable_weights=trainable_weights+list(loss_normalizer.parameters())
        

        assert data.shape[-1]==self.in_dim, f'Dataset dimension must match in_dim on model. data.shape[-1]({data.shape[-1]})!=model.in_dim({self.in_dim})'
        model.train()
        
        optim = torch.optim.AdamW(get_fm_optim_groups(model,loss_normalizer), lr=lr,fused=True)
        
        best_loss = float("inf")
        best_r2 = -1e8
        
        best_trained_model = deepcopy(model)
        
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs)
        
        prior_batch=torch.randn((batch_size,self.in_dim),device=device)
        time = torch.rand(batch_size,device=device)
        
        perm = torch.zeros(n, device=device,dtype=torch.int32)
        # model_trace = torch.jit.trace(model,example_inputs=(torch.randn((1,self.in_dim),device=device),torch.randn((1),device=device)))
        model_trace=model
        self.fit_history = {
            'loss':[],
            'r2':[]
        }
        
        try:
            for epoch in range(epochs):
                if debug and improved:
                    print(f"Epoch {epoch}: best_loss={best_loss:0.3f}\tbest r2={best_r2:0.3f}")
                improved = False
                
                # shuffle each epoch
                torch.randperm(n, device=device,out=perm)
                data_shuf = data[perm]
                condition_shuf = condition[perm]

                losses = 0
                r2s = 0
                for start in slices:
                    optim.zero_grad(set_to_none=True)  # set_to_none saves mem and can be faster [web:399]
                    
                    batch = data_shuf[start : start + batch_size]
                    B = batch.shape[0]
                    
                    zero_mask = (torch.rand(batch_size,device=device)<condition_dropout)[:B].unsqueeze(-1)
                    condition_batch = condition_shuf[start : start + batch_size]*zero_mask
                    
                    model_inference = lambda xt,t: model_trace(xt,t,condition_batch)
                    
                    # prior_batch.normal_()
                    prior_batch = sample_base(self.sobol,batch_size,device)
                    time.uniform_()
                    
                    pred_dir,target_dir,contrast_dir,t = \
                        model.fm.contrastive_flow_matching_pair(
                            model_inference,
                            prior_batch[:B],
                            batch,
                            time=time[:B]
                        )
                    
                    pred_loss = F.mse_loss(pred_dir,target_dir,reduction='none')+1
                    contrastive_loss = F.mse_loss(pred_dir,contrast_dir,reduction='none')
                    
                    contrastive_loss_det = contrastive_loss.detach()
                    pred_loss_det = pred_loss.detach()
                    # make it negative
                    contrastive_loss-=contrastive_loss_det.max()+1e-4
                    contrastive_loss=contrastive_loss/contrastive_loss_det.abs().mean()*pred_loss_det.abs().mean()
                    
                    # scale it
                    contrastive_loss = contrastive_loss_weight*contrastive_loss
                    
                    # sample-wise loss
                    sample_loss = pred_loss-contrastive_loss
                    
                    dm = (1-(1+epoch)/epochs)*distribution_matching
                    # dm=distribution_matching
                    if distribution_matching>0:
                        with torch.no_grad():  # Stop-gradient via detach
                            sg_log_losses = pred_loss_det.log()
                            target_log_w = -sg_log_losses  # log(1/L)
                        # dm = distribution_matching
                        weights = loss_normalizer(target_dir, t) # it equals to log(1/loss)
                        loss_weighted = (weights.detach()*dm).exp() # it equals to 1/loss
                        aux_loss = F.mse_loss(weights, target_log_w)
                    else:
                        loss_weighted=1
                        aux_loss=0
                    
                    #scale loss by it's prediction
                    weighed_loss = (loss_weighted*sample_loss).mean()
                    # print(r2_score(weights, (1/sample_loss).log()))
                    # print(weighed_loss)
                    loss = weighed_loss+dm*aux_loss
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1,
                        norm_type=2.0,
                    )
                        
                    optim.step()
                    
                    loss=loss.detach()
                    
                    r2 = r2_score(pred_dir,target_dir)
                    losses+=loss
                    r2s+=r2
                    
                    self.fit_history['loss'].append(loss.item())
                    self.fit_history['r2'].append(r2.item())
                    
                if scheduler: sch.step()
                
                mean_loss = losses/len(slices)
                mean_r2 = r2s/len(slices)
                if mean_r2 > best_r2:
                    best_loss = mean_loss
                    model_state_dict = model.state_dict()
                    best_trained_model = {key:model_state_dict[key].clone() for key in model_state_dict}
                    best_r2 = mean_r2
                    improved = True
        except KeyboardInterrupt:
            if debug: print("Stop training")
        finally:
            gc.enable()
            gc.collect()
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}\tbest_r2={best_r2:0.3f}")
        
        # update current model with best checkpoint
        model.load_state_dict(best_trained_model)
        model.eval()
    def __prepare_data(self, data, condition):
        if not isinstance(data,torch.Tensor):
            data = torch.tensor(data,dtype=torch.float32,device=self.device)
        if condition is not None and not isinstance(condition,torch.Tensor):
            condition = torch.tensor(condition,dtype=torch.float32,device=self.device)
        data = data.to(self.device).float()
        
        if self.conditional_dim is not None and condition is not None:
            condition = condition.to(self.device).float()
            if condition.ndim==1:
                condition=condition.unsqueeze(0)
            if condition.shape[0]==1:
                condition=condition[[0]*len(data)]
            assert len(condition)==len(data),'Dataset length and condition length must match'
            assert condition.shape[-1]==self.conditional_dim, f'Condition dimension must match conditional_dim on model. condition.shape[-1]({data.shape[-1]})!=model.conditional_dim({self.conditional_dim})'
        if condition is None:
            condition = torch.zeros((len(data),1))
        return data,condition
    def reflow(
            self,
            data : torch.Tensor,
            condition : Optional[torch.Tensor] = None,
            epochs = 512,
            steps : Literal[1,2] = 1,
            batch_size=256,
            debug = False,
            lr = 1e-2,
            weight_decay=0.01,
            distribution_matching = 0,
            grad_clip_max_norm : float|None=1,
            base_model : nn.Module|None = None,
            freeze_integrator = False
        ) -> None:
        """
        Distill a multi-step flow into a fast one-step or two-step generator.

        ReFlow trains the current model to directly approximate the transport
        map learned by a slower flow model. The resulting model can generate
        samples using only one or two transport evaluations while preserving
        the distribution learned by the teacher model.

        The procedure operates in both directions:

        - Prior -> Target
        - Target -> Prior

        allowing the distilled model to retain generation, inversion and
        density-estimation capabilities.

        During training a synthetic dataset is created by transporting samples
        through a teacher model. The distilled model is then trained to
        reproduce those mappings using a reduced number of integration steps.

        For conditional models the same conditioning vectors used by the
        teacher are propagated through the distillation process.

        Parameters
        ----------
        data : torch.Tensor
            Dataset sampled from the target distribution.

            Shape:

            ``[num_samples, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors associated with the dataset.

            Required for conditional models.

            Shape:

            ``[num_samples, conditional_dim]``

        epochs : int, default=512
            Number of ReFlow optimization iterations.

        steps : {1, 2}, default=1
            Target generator complexity.

            - ``1``: distill into a one-step generator
            - ``2``: distill into a two-step generator

            After successful training:

            ``self.default_steps = steps``

        batch_size : int, default=256
            Mini-batch size.

        debug : bool, default=False
            Print optimization statistics.

        lr : float, default=1e-2
            Learning rate.

        weight_decay : float, default=0.01
            Weight decay used by the optimizer.

        distribution_matching : float, default=0
            Enables adaptive loss reweighting.

            Larger values focus training on regions where the distilled model
            performs poorly.

            Typical values:

            - 0.0 : disabled
            - 0.05 - 0.25 : mild correction

        grad_clip_max_norm : float | None, default=1
            Maximum gradient norm.

            If None, gradient clipping is disabled.

        base_model : nn.Module | None, optional
            Teacher model.

            The teacher must implement:

            - ``to_target()``
            - ``to_prior()``

            If None, the current model is used as the teacher
            (self-distillation).

        freeze_integrator : bool, default=False
            Whether to freeze learned integration coefficients during ReFlow.

            If True:

            - one_weights
            - one_weights_inv
            - rk2_weights
            - rk2_weights_inv

            remain fixed.

            If False, the integrator coefficients are optimized jointly with
            the neural network.

        Returns
        -------
        None

        Notes
        -----
        ReFlow performs online dataset generation using the teacher model.

        The generated training set contains two components:

        1. Real dataset samples mapped into latent space.
        2. Synthetic samples generated by the teacher model from randomly
        sampled latent vectors.

        Mixing both sources improves coverage of latent space and reduces
        failures in poorly represented prior regions.

        Unlike standard Flow Matching training, ReFlow optimizes direct
        transport mappings:

        .. math::

            x \\rightarrow y

        and

        .. math::

            y \\rightarrow x

        rather than velocity-field prediction.

        Training statistics are stored in:

        .. code-block:: python

            self.reflow_history["loss"]
            self.reflow_history["forward_r2"]
            self.reflow_history["inverse_r2"]

        Examples
        --------
        Distill a trained flow into a one-step generator:

        >>> model.fit(data)
        >>> model.reflow(data, steps=1)

        Conditional ReFlow:

        >>> model.fit(data, condition)
        >>> model.reflow(data, condition, steps=1)

        Create a two-step distilled model:

        >>> model.reflow(data, steps=2)
        """

        self.unfreeze()
        gc.disable()
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
        except: pass
        if not isinstance(data,torch.Tensor):
            data = torch.tensor(data,dtype=torch.float32,device=self.device)
        self.to(self.device)
        
        data = data.to(self.device)
        if base_model is None: base_model=self
        if self.conditional_dim is not None:
            assert condition is not None,'Cannot reflow conditional model with None condition'
        if condition is None:
            condition = torch.zeros((len(data),self.conditional_dim or 1))
        condition = condition.to(self.device)
        
        with torch.no_grad():
            x = base_model.to_prior(data,condition)
            y = data
            
            # balance generated and original dataset 50/50
            # the thing is that dataset latent space may be too limited
            # to reach all edge-case samples from some subspaces of prior
            # and reflowed model may struggle to transport these subspace prior
            # samples to target distribution, so, we also include generated from base model
            # samples to reflow model training, this step empirically helps a lot
            # with reflowed model quality
            
            # x_gen = sample_base(self.sobol,len(x),self.device)
            x_gen = torch.randn_like(x)
            cond_gen = torch.zeros_like(condition)
            y_gen = base_model.to_target(x_gen,cond_gen)
            
            x = torch.concat([x,x_gen],0)
            y = torch.concat([y,y_gen],0)
            cond = torch.concat([condition,cond_gen],0)
            
        assert steps in [1,2],"steps parameter must be one of [1,2]"
        self.fm.reset_weights()
        self.train()
        self.default_steps=steps
        
        if freeze_integrator:
            self.fm.freeze()

        
        loss_normalizer = nn.Sequential(
            nn.Linear(self.in_dim*2,self.hidden_dim),
            Residual([
                nn.SiLU(),
                zero_module(nn.Linear(self.hidden_dim,self.hidden_dim)),
            ],init_at_zero=False),
            nn.RMSNorm(self.hidden_dim),
            Prod([
                nn.Linear(self.hidden_dim,self.hidden_dim),
                # nn.RMSNorm(self.hidden_dim),
                nn.Tanh(),
            ]),
            nn.SiLU(),
            nn.Linear(self.hidden_dim,2),
        ).to(self.device)
        
        device=self.device
        # loss_normalizer = torch.jit.trace(loss_normalizer,torch.randn((1,self.in_dim*2),device=self.device))
        # opt = torch.optim.AdamW(get_fm_optim_groups(self,loss_normalizer,weight_decay=weight_decay),lr=lr,fused=True)
        opt = torch.optim.AdamW(list(self.parameters())+list(loss_normalizer.parameters()),weight_decay=weight_decay,lr=lr,fused=True)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        mse = torch.nn.functional.mse_loss
        
        self.reflow_history = {
            'loss':[],
            'forward_r2':[],
            'inverse_r2':[],
        }
        # running_r2 = 0
        # best_r2 = 0
        # best_model = None
        # check_each=16
        try:
            for i in range(epochs):
                opt.zero_grad(True)
                ind = torch.randperm(x.shape[0],device=device)[:batch_size]
                xbatch = x[ind]
                ybatch = y[ind]
                condbatch = cond[ind]
                y_pred = self.to_target(xbatch,condbatch)
                x_pred = self.to_prior(ybatch,condbatch)
                
                forward_loss = mse(ybatch,y_pred,reduction='none').mean(-1)
                forward_loss-=forward_loss.min().detach()-1e-2
                inverse_loss = mse(xbatch,x_pred,reduction='none').mean(-1)
                inverse_loss-=inverse_loss.min().detach()-1e-2
                
                prediction_loss = forward_loss.mean()+inverse_loss.mean()
                
                forward_weight,inverse_weight = loss_normalizer(torch.concat([xbatch,ybatch],-1)).chunk(2,-1)
                forward_weight, inverse_weight = forward_weight[...,0], inverse_weight[...,0]
                normalizer_loss = mse(forward_weight,forward_loss.detach().log())+mse(inverse_weight,inverse_loss.detach().log())
                
                fw = (-forward_weight*distribution_matching).detach().exp()
                iw = (-inverse_weight*distribution_matching).detach().exp()
                loss = (fw*forward_loss).mean()+(iw*inverse_loss).mean()+normalizer_loss*distribution_matching
                
                forward_r2 = r2_score(ybatch,y_pred)
                inverse_r2 = r2_score(xbatch,x_pred)
                # running_r2 += (forward_r2+inverse_r2)/2
                # if (i+1)%check_each==0:
                #     if running_r2>best_r2:
                #         if debug: print("Save")
                #         best_r2=running_r2
                #         best_model=deepcopy(self.state_dict())
                #     running_r2=0
                    
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=grad_clip_max_norm,
                        norm_type=2.0,
                    )
                opt.step()
                sch.step()
                

                self.reflow_history['loss'].append(loss.item())
                self.reflow_history['forward_r2'].append(forward_r2.item())
                self.reflow_history['inverse_r2'].append(inverse_r2.item())
                if debug and (i+1)%32==0:
                    loss_pred_r2 = (r2_score(forward_weight,forward_loss.log())+r2_score(inverse_weight,inverse_loss.log()))/2
                    print(f"Iteration={(str(i)+" "*6)[:4]} loss={str(prediction_loss.detach().item())[:8]} forward_r2={str(forward_r2.item())[:6]} inverse_r2={str(inverse_r2.item())[:6]} loss_pred_r2={str(loss_pred_r2.item())[:6]}")
        except KeyboardInterrupt as e:
            print("Stop reflowing...")
        finally:
            gc.enable()
            gc.collect()
            
        # self.load_state_dict(best_model)
        self.eval()
    
    def to_prior(
        self,
        data: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps=None,
        return_intermediates=False
    ):
        """
        Transport samples from target space back into latent prior space.

        This is the inverse transport operator of the learned flow and is used
        internally for density estimation, interpolation, constrained
        optimization and latent-space editing.

        Parameters
        ----------
        data : torch.Tensor
            Samples from the target distribution.

            Expected shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors associated with the samples.

            Expected shape:

            ``[batch_size, conditional_dim]``

        steps : int | None, optional
            Number of integration steps.

            If None, ``self.default_steps`` is used.

        return_intermediates : bool, default=False
            Whether to return intermediate integration states.

        Returns
        -------
        torch.Tensor
            Corresponding latent vectors in prior space.

        tuple[torch.Tensor, list[torch.Tensor]]
            Returned when ``return_intermediates=True``.

            Contains:

            - Final latent vectors
            - Intermediate transport states

        Notes
        -----
        The returned latent vectors approximately follow a standard Gaussian
        distribution when the model is trained successfully.
        """

        if not steps: steps = self.default_steps
        input_device = data.device
        model_inference = lambda xt,t: self(xt,t,condition)
        data,condition=self.__prepare_data(data,condition)
        out = self.fm.integrate(model_inference,data,steps,inverse=True,return_intermediates=return_intermediates)
        if return_intermediates:
            out,inter = out
            out = out.to(input_device)
            inter = [l.to(input_device) for l in inter]
            out = (out,inter)
        else:
            out = out.to(input_device)
        return out
    def to_target(
        self,
        normal_noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps=None,
        return_intermediates=False
    ):
        """
        Transport samples from latent prior space into target data space.

        This is the forward transport operator of the learned flow. Samples
        from the Gaussian prior are iteratively transformed into samples from
        the target distribution using the learned velocity field.

        For conditional models, generation is conditioned on the supplied
        condition vectors.

        Parameters
        ----------
        normal_noise : torch.Tensor
            Samples from latent prior space.

            Expected shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors.

            Expected shape:

            ``[batch_size, conditional_dim]``

            If None and the model is conditional, zero-conditioning is used.

        steps : int | None, optional
            Number of integration steps.

            If None, ``self.default_steps`` is used.

        return_intermediates : bool, default=False
            Whether to return intermediate integration states.

        Returns
        -------
        torch.Tensor
            Generated samples in target space.

        tuple[torch.Tensor, list[torch.Tensor]]
            Returned when ``return_intermediates=True``.

            Contains:

            - Final generated samples
            - Intermediate transport states

        Notes
        -----
        After ReFlow distillation this method may become a one-step or two-step
        generator depending on the value of ``self.default_steps``.
        """

        if not steps: steps = self.default_steps
        input_device = normal_noise.device
        model_inference = lambda xt,t: self(xt,t,condition)
        normal_noise,condition=self.__prepare_data(normal_noise,condition)
        out = self.fm.integrate(model_inference,normal_noise,steps,return_intermediates=return_intermediates)
        if return_intermediates:
            out,inter = out
            out = out.to(input_device)
            inter = [l.to(input_device) for l in inter]
            out = (out,inter)
        else:
            out = out.to(input_device)
        return out
    def sample(
        self,
        num_samples : int,
        condition: Optional[torch.Tensor] = None,
        steps : Optional[int]=None,
        sobol : bool=False
    ):
        """
        Generate samples from the learned distribution.

        Samples are generated by drawing latent vectors from a Gaussian prior
        and transporting them into target space using ``to_target()``.

        Supports both unconditional and conditional generation.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        condition : torch.Tensor | None, optional
            Conditioning vectors.

            Expected shape:

            ``[num_samples, conditional_dim]``

            If None and the model is conditional, zero-conditioning is used.

        steps : int | None, optional
            Number of transport steps.

            If None, ``self.default_steps`` is used.

        sobol : bool, default=False
            Use Sobol low-discrepancy sampling instead of standard Gaussian
            sampling.

            Sobol sampling often improves latent-space coverage and may reduce
            variance for small sample counts.

        Returns
        -------
        torch.Tensor
            Generated samples.

            Shape:

            ``[num_samples, in_dim]``
        """

        if not steps: steps = self.default_steps
        if sobol:
            x = sample_base(self.sobol,num_samples,self.device)
        else:
            x = torch.randn((num_samples,self.in_dim),device=self.device)
        return self.to_target(x,condition,steps=steps)
    def constrained_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        num_samples: int,
        condition : Optional[torch.Tensor] = None,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 0.0,
        sampler_steps = None
    ) -> torch.Tensor:
        """
        Generate samples satisfying arbitrary differentiable constraints.

        The optimization is performed in latent space rather than data space.
        Latent vectors are adjusted using LBFGS while balancing:

        - Constraint satisfaction
        - Prior probability preservation
        - Optional mode-seeking behavior

        This method is particularly useful for inverse design problems where
        generated samples must satisfy user-defined objectives.

        Parameters
        ----------
        constraint : Callable[[torch.Tensor], torch.Tensor]
            Differentiable constraint function.

            Receives generated samples:

            ``[batch_size, in_dim]``

            Returns a scalar loss.

        num_samples : int
            Number of samples to generate.

        condition : torch.Tensor | None, optional
            Conditioning vectors.

        noise_scale : float, default=0.0
            Langevin-style noise added after optimization steps.

            Small values can improve exploration.

            Typical range:

            ``0.0 - 0.05``

        steps : int, default=2
            Number of latent optimization iterations.

        lr : float, default=1
            LBFGS learning rate.

        mode_closeness_weight : float, default=0.0
            Additional penalty encouraging solutions closer to latent-space
            modes.

            Large values may cause mode collapse.

        sampler_steps : int | None, optional
            Number of flow integration steps used during optimization.

            If None, ``self.default_steps`` is used.

        Returns
        -------
        torch.Tensor
            Generated samples satisfying the constraint as closely as possible.
        """
        model = self
        model.eval()
        self.freeze()
        device = self.device
        # Initialize z from standard normal distribution
        z = sample_base(self.sobol,num_samples,device=device).requires_grad_(True)
        original_prior = (z * z).mean().detach()

        # Create optimizer for the latent variable z
        optimizer = torch.optim.LBFGS([z], lr=lr)

        class Iteration:
            best_sample = z.clone().detach()
            best_loss = 1e8
        self._iteration = Iteration()
        
        def closure():
            optimizer.zero_grad()

            # Forward pass: x = M_inv(z)
            x = model.to_target(z,condition,steps=sampler_steps)

            # Balance original prior probability/vs likelihood maximization
            L_prior = (z * z).mean()
            L_prior = (L_prior-original_prior)**2+mode_closeness_weight*L_prior

            # Compute constraint loss: L_constraint = constraint(x)
            L_constraint = constraint(x)

            # Total loss: L_total = L_prior + λ * L_constraint
            L_total = L_prior + L_constraint

            it = self._iteration
            if L_total<it.best_loss:
                it.best_loss = L_total
                it.best_sample = z.clone().detach()
            
            L_total.backward()
            with torch.no_grad():
                z.data += noise_scale * torch.randn_like(z)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.to_target(self._iteration.best_sample,condition,steps=sampler_steps)
        self.unfreeze()
        return final_x
    def constrained_optimize(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        data,
        condition : Optional[torch.Tensor] = None,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 0.0,
        sampler_steps = None
    ) -> torch.Tensor:
        """
        Optimize existing samples subject to a differentiable constraint.

        Unlike ``constrained_sample()``, this method starts from existing data,
        maps it into latent space, performs optimization there, and then maps
        the optimized latent vectors back into target space.

        The optimization attempts to preserve original sample probability while
        minimizing the supplied constraint.

        Parameters
        ----------
        constraint : Callable[[torch.Tensor], torch.Tensor]
            Differentiable objective function.

        data : torch.Tensor
            Initial samples.

            Shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors.

        noise_scale : float, default=0.0
            Langevin-style exploration noise.

        steps : int, default=2
            Number of optimization iterations.

        lr : float, default=1
            LBFGS learning rate.

        mode_closeness_weight : float, default=0.0
            Additional mode-seeking regularization.

        sampler_steps : int | None, optional
            Number of transport steps used during optimization.

        Returns
        -------
        torch.Tensor
            Optimized samples in target space.
        """
        model = self
        model.eval()
        self.freeze()
        device = self.device
        # Move data to prior
        with torch.no_grad():
            z : torch.Tensor = self.to_prior(data,condition,steps=sampler_steps)
        z=z.requires_grad_(True)
        
        original_prior = (z * z).mean().detach()

        # Create optimizer for the latent variable z
        optimizer = torch.optim.LBFGS([z], lr=lr)

        class Iteration:
            best_sample = z.clone().detach()
            best_loss = 1e8
        self._iteration = Iteration()
        
        def closure():
            optimizer.zero_grad()

            # Forward pass: x = M_inv(z)
            x = model.to_target(z,condition,steps=sampler_steps)

            # Balance original prior probability/vs likelihood maximization
            L_prior = (z * z).mean()
            L_prior = (L_prior-original_prior)**2+mode_closeness_weight*L_prior

            # Compute constraint loss: L_constraint = constraint(x)
            L_constraint = constraint(x)

            # Total loss: L_total = L_prior + λ * L_constraint
            L_total = L_prior + L_constraint

            it = self._iteration
            if L_total<it.best_loss:
                it.best_loss = L_total
                it.best_sample = z.clone().detach()
            
            L_total.backward()
            with torch.no_grad():
                z.data += noise_scale * torch.randn_like(z)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.to_target(self._iteration.best_sample,condition,steps=sampler_steps)
        self.unfreeze()
        return final_x
    def optimize(
        self, 
        data: torch.Tensor,
        condition : Optional[torch.Tensor] = None,
        lr: float = 1.0, 
        epochs: int = 1,
        columns_to_optimize: list[int] = None,
        random_directions=0
    ):
        """
        Optimize specific columns of data to maximize log probability.

        This method performs gradient-based optimization to adjust specific columns of the input
        data to increase their likelihood under the learned model distribution. It keeps other
        columns fixed while optimizing the specified ones.

        Args:
            data (torch.Tensor): Input tensor of shape [batch_size, input_dim] to optimize
            lr (float): Learning rate for the LBFGS optimizer (default: 1.0)
            epochs (int): Number of optimization epochs (default: 1)
            columns_to_optimize (list[int]): List of column indices to optimize (0-based).
                                           If None or empty, all columns will be optimized.
            random_directions: log-prob random directions approximation vectors
        Returns:
            tuple: A tuple containing:
                 - torch.Tensor: Optimized data tensor with the same shape as input
                 - torch.Tensor: Final loss value after optimization
        """
        batch_size, input_dim = data.shape
        self.freeze()
        # Handle default case - optimize all columns if none specified
        if columns_to_optimize is None or len(columns_to_optimize) == 0:
            columns_to_optimize = list(range(input_dim))

        # Validate column indices
        columns_to_optimize = [c for c in columns_to_optimize if 0 <= c < input_dim]
        if not columns_to_optimize:
            return data.clone(), -self.log_prob(data,condition=condition,random_directions=random_directions).sum().detach()

        # Identify fixed columns as those not in columns_to_optimize
        all_columns = list(range(input_dim))
        fixed_columns = [c for c in all_columns if c not in columns_to_optimize]

        # Create optimizable parameters for only the specified columns
        optimizable_data = data[:, columns_to_optimize].clone().detach().requires_grad_(True)

        # Fixed data doesn't need gradients
        if fixed_columns:
            fixed_data = data[:, fixed_columns].clone().detach()
        else:
            fixed_data = None

        # Define optimizer on only the optimizable part
        optimizer = torch.optim.LBFGS([optimizable_data], lr=lr, max_iter=20)

        class IterationData:
            best_loss = 1e8
            best_optimizable_data = optimizable_data.clone().detach()

        iteration = IterationData()
        self._iteration = iteration
        # Reconstruct full tensor by combining optimizable and fixed parts
        self._current_data = torch.zeros_like(data)

        def closure():
            optimizer.zero_grad(True)

            iteration = self._iteration
            current_data = self._current_data.detach()

            # Fill in the optimizable columns
            current_data[:, columns_to_optimize] = optimizable_data

            # Fill in fixed columns if any exist
            if fixed_columns:
                current_data[:, fixed_columns] = fixed_data

            # Compute loss on the full tensor
            loss = -self.log_prob(current_data,condition=condition,random_directions=random_directions).sum()

            if loss<iteration.best_loss:
                iteration.best_loss=loss
                iteration.best_optimizable_data=optimizable_data.detach().clone()

            loss.backward()

            return loss

        # Run optimization
        for i in range(epochs):
            loss = optimizer.step(closure)

        # Create final result by combining optimized and fixed parts
        result = torch.zeros_like(data)
        result[:, columns_to_optimize] = iteration.best_optimizable_data

        # Add back fixed columns if any exist
        if fixed_columns:
            result[:, fixed_columns] = fixed_data
        self.unfreeze()
        return result, iteration.best_loss
    def log_prob(
        self, 
        data, 
        condition:Optional[torch.Tensor] = None,
        eps=1e-3,
        random_directions=0,
        return_prior=False):
        """
        Estimate log-probability under the learned distribution.

        Density estimation is performed by transporting samples into latent
        space and estimating the inverse-flow Jacobian determinant.

        Supports conditional densities when condition vectors are supplied.

        Parameters
        ----------
        data : torch.Tensor
            Samples for density evaluation.

            Shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors.

        eps : float, default=1e-3
            Finite-difference step size used for Jacobian estimation.

        random_directions : int, default=0
            Number of random projection directions used for Jacobian
            approximation.

            Values:

            - 0 : exact directional evaluation
            - >0 : stochastic approximation

            Larger values are typically faster for high-dimensional problems.

        return_prior : bool, default=False
            Forwarded to ``log_prob_inverse``.

        Returns
        -------
        torch.Tensor
            Estimated log-probabilities.

        Notes
        -----
        For conditional models the returned density corresponds to:

        .. math::

            \\log p(x \\mid c)

        rather than the unconditional density.
        """
        
        to_prior = lambda xt:self.to_prior(xt,condition)
        # return log_prob(self.to_target,self.to_prior(data),eps,random_directions=random_directions)
        return log_prob_inverse(to_prior,data.to(self.device),eps,random_directions=random_directions,return_prior=return_prior)
    def freeze(self):
        """
        Disables grad on model weights
        """
        for p in self.parameters():
            p.requires_grad_(False)
    def unfreeze(self):
        """
        Enables grad on model weights
        """
        for p in self.parameters():
            p.requires_grad_(True)
    def interpolate(
        self,
        A:torch.Tensor,
        B:torch.Tensor,
        t:torch.Tensor|float,
        A_condition : Optional[torch.Tensor] = None,
        B_condition : Optional[torch.Tensor] = None):
        """
        Interpolate between samples through the learned latent space.

        Samples are first mapped into latent space, interpolated linearly,
        and then mapped back into target space.

        Compared to direct interpolation in data space, latent interpolation
        often produces smoother and more realistic trajectories.

        Conditional interpolation is also supported.

        When both condition vectors are supplied, conditions are interpolated
        together with latent representations.

        Parameters
        ----------
        A : torch.Tensor
            Starting samples.

            Shape:

            ``[batch_size, in_dim]``

        B : torch.Tensor
            Ending samples.

            Shape:

            ``[batch_size, in_dim]``

        t : float | torch.Tensor
            Interpolation locations.

            Values must lie in:

            ``[0, 1]``

            Examples:

            .. code-block:: python
                t = 0.5
                t = torch.linspace(0, 1, 128)

        A_condition : torch.Tensor | None, optional
            Conditions associated with A.

        B_condition : torch.Tensor | None, optional
            Conditions associated with B.

        Returns
        -------
        torch.Tensor
            Interpolated samples.

        Notes
        -----
        The interpolation is performed as:

        1. A -> latent space
        2. B -> latent space
        3. Linear interpolation in latent space
        4. Transport back into target space

        If both conditions are supplied:

        .. math::

            c_t = (1-t)c_A + tc_B

        is used during decoding.
        """
        
        if isinstance(t,float):t=torch.tensor([t])

        if A.ndim==1:A=A.unsqueeze(0)
        if B.ndim==1:B=B.unsqueeze(0)
        if t.ndim==1:t=t.unsqueeze(1).unsqueeze(1)
        if A_condition is not None:
            if A_condition.ndim==1:A_condition=A_condition.unsqueeze(0)
        
        if B_condition is not None:
            if B_condition.ndim==1:B_condition=B_condition.unsqueeze(0)

        with torch.no_grad():
            A_prior = self.to_prior(A,A_condition)
            B_prior = self.to_prior(B,B_condition)

            prior_interp = torch.lerp(A_prior,B_prior,t)
            if A_condition is not None and B_condition is not None:
                condition_interp = torch.lerp(A_condition,B_condition,t)
            else:
                condition_interp=None
            AB_interp = self.to_target(prior_interp,condition_interp)
        
        return AB_interp
        
