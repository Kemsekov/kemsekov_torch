from copy import deepcopy
import gc
import math
import os
from typing import Callable, Literal, Optional
from kemsekov_torch.common_modules import Prod, Residual
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.common_modules import mmd_rbf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.func import vmap, jacrev

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
    ts = time_transform(ts[:,None])[:,0]
    
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
    device = list(model.parameters())[0].device
    if inverse:
        ts = torch.linspace(1, 0, steps+1, device=device)  # steps intervals = steps+1 points
    else:
        ts = torch.linspace(0, 1, steps+1, device=device)
    ts = time_transform(ts[:,None])[:,0]
    dt = ts[1]-ts[0]
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
            x_pred = xt + dt * pred_current
            
            # Evaluate at predicted point
            t_expand_next = t_next.expand(x0.shape[0])
            pred_next = pred(x_pred, t_expand_next)
            
            # Corrector step (Heun's method)
            xt = xt + dt * 0.5 * (pred_current + pred_next)
            
            # Store the predictor derivative for next step
            prev_pred = pred_next
        else:
            # Subsequent steps: reuse previous derivative (1 evaluation per step)
            # Predictor using previous derivative (Euler step)
            x_pred = xt + dt * prev_pred
            
            # Evaluate at predicted point (ONLY ONE EVAL PER STEP)
            t_expand_next = t_next.expand(x0.shape[0])
            pred_next = pred(x_pred, t_expand_next)
            
            # Corrector step using stored previous derivative
            xt = xt + dt * 0.5 * (prev_pred + pred_next)
            
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

def rk2(model, x0, weights, return_intermediates=False):
    device = next(model.parameters()).device
    x0 = x0.to(device)
    xt = x0.clone()
    
    
    left = weights[0]
    dt = weights[1]
    weight_k1 = weights[2]
    weight_k2 = weights[3]
    model_eval_t = weights[4]
    
    t_start =   left
    
    intermediates = []
    
    # === First evaluation (k1 at start) - weight = 1/4 ===
    
    k1 = model(xt, model_eval_t.unsqueeze(0))
    
    # === Second evaluation (k2 at 2/3 point) - Ralston's optimal point ===
    t_ralston = t_start + weights[4] * dt  # = 1/3 for reverse, 2/3 for forward
    x_ralston = xt + weights[5] * dt * k1
    t_expand_ralston = t_ralston.expand(x0.shape[0])
    k2 = model(x_ralston, t_expand_ralston)
    
    xt_next = xt + dt * (weight_k1 * k1 + weight_k2 * k2)
    
    if return_intermediates:
        intermediates.extend([x_ralston, xt_next])
    
    return (xt_next, intermediates) if return_intermediates else xt_next

def one_step(model,x0,weights):
    """One-step integration"""
    t=weights[0].unsqueeze(0)
    pred = model(x0,t)
    # add bias term
    return weights[1]*pred+weights[2]*x0+weights[3]*x0*x0
class FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_scaler = lambda x:x
        self.time_sampler_transform = lambda x:x
        self.reset_weights()
        # weights for one-step integration
    
    def reset_weights(self):
        with torch.no_grad():
            self.one_weights     = torch.nn.Parameter(torch.tensor([0.5,  0.5, 1,0,0]))
            self.one_weights_inv = torch.nn.Parameter(torch.tensor([0.5,-0.5,1,0,0]))
            self.rk2_weights     = torch.nn.Parameter(torch.tensor([0,1,1/4,3/4,2/3,2/3,0.0]))
            self.rk2_weights_inv = torch.nn.Parameter(torch.tensor([1,-1,1/4,3/4,2/3,2/3,1.0]))
    
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
        if isinstance(steps,torch.Tensor):
            steps=steps.int().item()
        match steps:
            case 1: return one_step(model,x0,self.one_weights_inv if inverse else self.one_weights)
            case 2: return rk2(model,x0,self.rk2_weights_inv if inverse else self.rk2_weights,return_intermediates)
            case 3: return rk3(model,x0,churn_scale,inverse,return_intermediates)
            case _: return heun(model,x0,steps-1,churn_scale,inverse,return_intermediates,time_transform=self.time_sampler_transform,no_grad_model=no_grad_model)

def generate_unit_simplex_vertices(d):
    # 1. Generate the initial regular simplex
    vertices = torch.eye(d)
    last_vertex = (1 - torch.sqrt(torch.tensor(d + 1.0))) / d * torch.ones(d)
    vertices = torch.cat([vertices, last_vertex.unsqueeze(0)], dim=0)
    
    # 2. Recenter at the origin (subtract the mean)
    vertices -= vertices.mean(dim=0)
    
    # 3. Normalize to unit length (radius = 1)
    # Each row is a vertex; normalize along the last dimension
    return torch.nn.functional.normalize(vertices, p=2, dim=-1)

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
        m = Residual([
            Prod(nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.RMSNorm(hidden_dim),
                nn.Tanh(),
            )),
            nn.SiLU(),
            zero_module(nn.Linear(hidden_dim,hidden_dim)),
        ],init_at_zero=False)
        prod = m.m[0]
        self.prod_linear = prod.module[0]
        self.prod_norm = prod.module[1]
        self.linear = m.m[2]
        self.m=m
    def forward(self,x):
        # i have tested it, it returns exactly same result as self.m(x)
        pl = self.prod_linear
        pn = self.prod_norm
        ln = self.linear
        x_orig = x
        x1 = F.linear(x,pl.weight,pl.bias)
        x = x*F.rms_norm(x1,pn.normalized_shape,pn.weight).tanh_()
        x = F.linear(F.silu(x,inplace=True),ln.weight,ln.bias)+x_orig
        return x


class FlowModel1d(nn.Module):
    """
    Flow-matching model for 1-dimensional (vector) data.

    This class implements a flow matching model that learns to transform simple distributions
    (like Gaussian noise) into complex target distributions (like data samples). It supports
    training, sampling, conditional sampling, and probability computation.

    The model uses residual blocks with time embeddings to predict the vector field that
    describes the flow between distributions. It includes methods for training with contrastive
    flow matching, sampling from the learned distribution, and computing log probabilities.

    Attributes:
        fm (FlowMatching): Flow matching instance for handling core flow operations
        in_dim (int): Input dimension of the data
        hidden_dim (int): Hidden dimension of the neural network
        time_emb (nn.Sequential): Time embedding network
        expand (nn.Linear): Linear layer to expand input to hidden dimension
        dropout (nn.Dropout): Dropout layer for regularization
        residual_blocks (nn.ModuleList): List of residual blocks with attention mechanisms
        collapse (nn.Linear): Linear layer to collapse hidden dimension back to input dimension
        default_steps (int): Default number of integration steps for sampling
        device (str): Device on which the model is located
    """
    def __init__(self, in_dim,hidden_dim=64,residual_blocks=3,dropout_p=0.0,device='cpu') -> None:
        super().__init__()
        self.fm = FlowMatching()
        # default time scaler for training
        self.fm.time_scaler = lambda x: torch.log(9*x+1)/math.log(10)
        
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        norm = nn.RMSNorm

        
        self.time_emb = nn.Sequential(
            nn.Linear(1,hidden_dim),
            Prod(nn.Sequential(
                nn.RMSNorm(hidden_dim),
                nn.Linear(hidden_dim,hidden_dim),
                nn.Tanh(),
            )),
            nn.SiLU(),
            zero_module(nn.Linear(hidden_dim,hidden_dim*2)),
        )
        
        self.expand = nn.Linear(in_dim,hidden_dim)
        
        self.dropout = nn.Dropout(dropout_p) if dropout_p>0 else nn.Identity()
        self.norm = norm(hidden_dim)
        
        self.residual_blocks = nn.ModuleList([
            FusedFlowResidual(hidden_dim)
            for i in range(residual_blocks)
        ])
        
        self.collapse = nn.Sequential(
            norm(hidden_dim),
            nn.Linear(hidden_dim,in_dim)
        )
        # self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.default_steps=16
        self.to(device)
        self.eval()

        
    def forward(self,x : torch.Tensor,t : torch.Tensor):
        # x_orig = x
        while t.ndim<x.ndim:
            t = t[:,None]
        time = self.time_emb(t)
        expand = self.expand(x)
        
        # add time embedding
        time_scale,time_shift = time.chunk(2,-1)
        x = expand*(1+time_scale)+time_shift
        
        x = self.dropout(x)
        x = self.norm(x)
        for m in self.residual_blocks:
            x = m(x)
        
        return self.collapse(x)#+x_orig*self.gamma
    
    def to(self,device):
        """
        Moves the model to the specified device.

        This method overrides the parent's to() method to also update the device attribute
        of the FlowModel1d instance, ensuring that the model knows which device it's on.

        Args:
            device: The device (CPU/GPU) to move the model to

        Returns:
            FlowModel1d: The model moved to the specified device
        """
        self.device=device
        return super().to(device)
    def fit(
        self,
        data: torch.Tensor,
        batch_size: int = 256,
        epochs: int = 64,
        contrastive_loss_weight=0.1,
        lr: float = 0.02,
        distribution_matching=0.0,
        debug: bool = False,
        scheduler = True,
    ):
        """
        Trains the flow matching model on the provided data using contrastive flow matching.

        This method implements a sophisticated training procedure that includes:
        - Contrastive flow matching with negative samples
        - Dynamic loss normalization using a separate neural network
        - Optional reflow training with custom prior datasets
        - Gradient clipping for stable training
        - Model checkpointing to retain the best performing version

        The training alternates between optimizing the main flow model and the loss normalizer,
        using a weighted loss that combines MSE between predictions and targets with a
        contrastive term that pushes predictions away from negative samples.

        Args:
            data (torch.Tensor): Training data tensor of shape [N, input_dim].
            batch_size (int): Number of samples per training batch (default: 512).
            epochs (int): Number of training epochs (default: 100).
            contrastive_loss_weight (float): Weight for the contrastive loss term that
                                           uses negative samples (default: 0.1).
            lr (float): Learning rate for the AdamW optimizer (default: 0.02).
            normalizer_loss_weight (float): Weight for the auxiliary loss that trains
                                          the loss normalizer network (default: 0.1).
            distribution_matching: how close to original data distributed resulting flow must be. When set to 1, model will try to match distribution exactly, when close to 0, will shift distribution closer to mode. I advice you left it with 0.0.
            debug (bool): Whether to print training progress and best loss updates (default: False).
            scheduler (bool): Whether to use a cosine annealing learning rate scheduler (default: True).

        Returns:
            None: Modifies the model in-place, retaining the best checkpoint based on validation loss.
        """
        # these are optimal for cpu-training
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
        except: pass
        gc.disable()
        model = self
        device = model.device
        
        
        trainable_weights = list(model.parameters())
        if distribution_matching>0:
            loss_normalizer = LossNormalizer1d(model.in_dim,model.hidden_dim).to(device)
            trainable_weights=trainable_weights+list(loss_normalizer.parameters())
        
        batch_size = min(batch_size,data.shape[0])
        
        
        data = data.to(device)
        
        model.train()
        
        optim = torch.optim.AdamW(trainable_weights, lr=lr,fused=True)
        
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
        # model_trace = torch.jit.trace(model,example_inputs=(torch.randn((batch_size,self.in_dim)),torch.randn((batch_size))))
        
        try:
            for epoch in range(epochs):
                if debug and improved:
                    print(f"Epoch {epoch}: best_loss={best_loss:0.3f}\tbest r2={best_r2:0.3f}")
                improved = False
                
                # shuffle each epoch
                torch.randperm(n, device=device,out=perm)
                data_shuf = data[perm]

                losses = 0
                r2s = 0
                for start in slices:
                    optim.zero_grad(set_to_none=True)  # set_to_none saves mem and can be faster [web:399]
                    
                    batch = data_shuf[start : start + batch_size]
                    prior_batch.normal_()
                    time.uniform_()
                    
                    
                    B = batch.shape[0]
                    pred_dir,target_dir,contrast_dir,t = \
                        model.fm.contrastive_flow_matching_pair(
                            model,
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
                    
                    if distribution_matching>0:
                        with torch.no_grad():  # Stop-gradient via detach
                            sg_log_losses = pred_loss_det.log()
                            target_log_w = -sg_log_losses  # log(1/L)
                        
                        weights = loss_normalizer(target_dir, t) # it equals to log(1/loss)
                        loss_weighted = (weights.detach()*distribution_matching).exp() # it equals to 1/loss
                        aux_loss = F.mse_loss(weights, target_log_w)
                    else:
                        loss_weighted=1
                        aux_loss=0
                    
                    #scale loss by it's prediction
                    weighed_loss = (loss_weighted*sample_loss).mean()
                    # print(r2_score(weights, (1/sample_loss).log()))
                    # print(weighed_loss)
                    loss = weighed_loss+distribution_matching*aux_loss
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1,
                        norm_type=2.0,
                    )
                        
                    optim.step()
                    
                    r2 = r2_score(pred_dir,target_dir)
                    losses+=loss.detach()
                    r2s+=r2
                    
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
    
    def mmd2_with_data(self,data : torch.Tensor) -> float:
        """
        Computes the Maximum Mean Discrepancy (MMD) squared between the model's samples and given data.

        This method generates samples from the trained model and compares them to the provided data
        using the RBF kernel-based MMD metric. Lower values indicate that the model's samples
        are more similar to the provided data, suggesting better model performance.

        Args:
            data (torch.Tensor): Reference data tensor to compare against model samples.
                               Expected shape: [num_samples, input_dim]

        Returns:
            float: MMD^2 value indicating the discrepancy between model samples and reference data.
                   Lower values indicate better similarity between the distributions.
        """
        with torch.no_grad():
            sampled = self.sample(len(data))
            return mmd_rbf(data.to(self.device),sampled)[0].item()
    
    def reflow(
            self,
            data : torch.Tensor,
            steps : Literal[1,2] = 1,
            batch_size=256,
            iterations = 512,
            debug = False,
            lr = 1e-2,
            distribution_matching = 0,
            grad_clip_max_norm : float|None=1,
            base_model : nn.Module|None = None
        ) -> None:
        """
        Distill a teacher flow model into one-step generation via bidirectional ReFlow training.
        
        This method performs online knowledge distillation by training the current model (`self`) 
        to match a teacher model's (`base_model.to_target/to_prior`) one-step mappings using:
        
        Args:
            data (torch.Tensor): Target data tensor of shape [N, input_dim] from teacher distribution
            batch_size (int): Mini-batch size for training (default: 512)
            iterations (int): Total training iterations
            debug (bool): Print training progress and R² metrics (default: False)
            base_model (Optional[nn.Module]): Teacher model with `to_prior()`/`to_target()` methods. 
                                            If None, uses `self` (self-distillation, default: None)
        """
        gc.disable()
        if base_model is None: base_model=self
        with torch.no_grad():
            x = base_model.to_prior(data)
            y = data
        assert steps in [1,2],"steps parameter must be one of [1,2]"
        self.fm.reset_weights()
        self.train()
        self.default_steps=steps

        
        loss_normalizer = nn.Sequential(
            nn.Linear(self.in_dim*2,self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            Residual([
                nn.SiLU(),
                zero_module(nn.Linear(self.hidden_dim,self.hidden_dim)),
            ],init_at_zero=False),
            Prod(
                nn.Sequential(
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Tanh(),
                )
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim,2),
        )
        
        opt = torch.optim.AdamW(list(self.parameters())+list(loss_normalizer.parameters()),lr=lr,fused=True)
        mse = torch.nn.functional.mse_loss
        try:
            for i in range(iterations):
                opt.zero_grad(True)
                ind = torch.randperm(x.shape[0])[:batch_size]
                xbatch = x[ind]
                ybatch = y[ind]
                
                y_pred = self.to_target(xbatch)
                x_pred = self.to_prior(ybatch)
                
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
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=grad_clip_max_norm,
                        norm_type=2.0,
                    )
                opt.step()
                
                if debug:
                    loss_pred_r2 = (r2_score(forward_weight,forward_loss.log())+r2_score(inverse_weight,inverse_loss.log()))/2
                    print(f"Iteration={(str(i)+" "*6)[:4]} loss={str(prediction_loss.detach().item())[:8]} forward_r2={str(r2_score(ybatch,y_pred).item())[:6]} inverse_r2={str(r2_score(xbatch,x_pred).item())[:6]} loss_pred_r2={str(loss_pred_r2.item())[:6]}")
        except KeyboardInterrupt as e:
            print("Stop reflowing...")
        finally:
            gc.enable()
            gc.collect()
            
        self.eval()
  
    def to_prior(self,data : torch.Tensor,steps=None):
        """
        Transforms data from the target distribution back to the prior (noise) distribution.

        This method performs the inverse transformation, mapping samples from the complex
        target distribution (e.g., images, time series) back to the simple prior distribution
        (typically Gaussian noise). This is achieved by integrating the flow in reverse.

        Args:
            data (torch.Tensor): Input tensor from the target distribution.
                               Expected shape: [batch_size, input_dim]
            steps (int, optional): Number of integration steps to use.
                                 If None, uses the model's default steps.

        Returns:
            torch.Tensor: Transformed tensor in the prior distribution  ace.
                         Same shape as input tensor.
        """
        if not steps: steps = self.default_steps
        input_device = data.device
        return self.fm.integrate(self,data.to(self.device),steps,inverse=True).to(input_device)
    
    def to_target(self,normal_noise : torch.Tensor,steps=None):
        """
        Transforms samples from the prior (noise) distribution to the target distribution.

        This method performs the forward transformation, mapping samples from the simple
        prior distribution (typically Gaussian noise) to the complex target distribution
        (e.g., images, time series). This is achieved by integrating the learned flow.

        Args:
            normal_noise (torch.Tensor): Input tensor from the prior distribution.
                                       Expected shape: [batch_size, input_dim]
            steps (int, optional): Number of integration steps to use.
                                 If None, uses the model's default steps.

        Returns:
            torch.Tensor: Transformed tensor in the target distribution space.
                         Same shape as input tensor.
        """
        if not steps: steps = self.default_steps
        input_device = normal_noise.device
        return self.fm.integrate(self,normal_noise.to(self.device),steps).to(input_device)
    
    def sample(self,num_samples,steps=None):
        """
        Generates samples from the learned target distribution.

        This method creates new samples by first generating random Gaussian noise and then
        transforming it through the learned flow to the target distribution. This is the
        primary method for generating new data from the trained model.

        Args:
            num_samples (int): Number of samples to generate
            steps (int, optional): Number of integration steps to use for transformation.
                                 If None, uses the model's default steps.

        Returns:
            torch.Tensor: Generated samples from the target distribution.
                         Shape: [num_samples, input_dim]
        """
        if not steps: steps = self.default_steps
        return self.to_target(torch.randn((num_samples,self.in_dim),device=self.device),steps)
    
    def conditional_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        num_samples: int,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 1.0,
        sampler_steps = None
    ) -> torch.Tensor:
        """
        Make conditional sampling of trained flow matching model.
        
        I **strongly** advice you to call `reflow(...)` method before using conditional sampling,
        otherwise you will need a lot more time to execute this method.
        
        Args:
            constraint: Constraint loss function. Accepts generated target in `(num_samples,dim)` shape and returns loss `(scalar tensor)` that defines condition for sampling.
            num_samples: Number of samples to generate
            noise_scale: Scale of noise added during Langevin dynamics (default 0.00). Increasing this value will result in samples more spread from condition. Values around [0 to 0.05] are generally good enough.
            steps: Number of optimization steps (default 2)
            lr: Learning rate for the optimization (default 1)
            mode_closeness_weight: Weight for trying to sample closer to distribution mode. Increasing this value make samples cluster more around closest distribution mode, potentially leading to mode collapse (all samples are the same).
            sampler_steps: sampler steps for flow matching models.
        Returns:
            torch.Tensor: Samples of shape `[num_samples, input_dim]` satisfying the conditions
        
        """
        model = self
        model.eval()

        # Initialize z from standard normal distribution
        z = torch.randn(num_samples, model.in_dim, device=self.device, requires_grad=True)

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
            x = model.to_target(z,sampler_steps)

            # Compute prior loss: L_prior = ||z||² (keep z in N(0,I)) must match original generated prior
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
                z.data += (noise_scale) * torch.randn_like(z)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.to_target(self._iteration.best_sample)

        return final_x
    
    def full_log_prob(self, data: torch.Tensor,steps=None) -> torch.Tensor:
        """
        Computes the full log probability of the data using exact Jacobian computation.

        This method calculates the exact log probability using the change-of-variables formula:
        log p(x) = log p(z) + log |det(J)|, where z is the corresponding latent variable
        and J is the Jacobian of the transformation. This is computationally expensive
        due to the Jacobian determinant calculation, so the log_prob method is recommended
        for most use cases.

        Args:
            data (torch.Tensor): Input tensor for which to compute log probability.
                               Expected shape: [batch_size, input_dim]
            steps (int, optional): Number of integration steps to use for transformation.
                                 If None, uses the model's default steps.

        Returns:
            torch.Tensor: Log probability values for each sample in the batch.
                         Shape: [batch_size,]
        """
        if not steps: steps = self.default_steps

        Y = data.to(self.device) # complex domain
        X = self.to_prior(Y,steps) # normal noise
        def elementwise_prior(x): return self.to_prior(x,steps).view(-1,dim).sum(0)

        # change of variables stuff
        dim = Y.shape[-1]
        Y_flat = Y.view(-1,dim)
        output_shape = *Y.shape[:-1],dim,dim
        batch_jac = vmap(jacrev(elementwise_prior),randomness='same')(Y_flat).view(output_shape)
        log_jac_det = batch_jac.slogdet()[1]
        prior_log_prob = Normal(0,1).log_prob(X).sum(-1)

        return prior_log_prob+log_jac_det
    
    def optimize(self, data: torch.Tensor, lr: float = 1.0, epochs: int = 1,
             columns_to_optimize: list[int] = None):
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

        Returns:
            tuple: A tuple containing:
                 - torch.Tensor: Optimized data tensor with the same shape as input
                 - torch.Tensor: Final loss value after optimization
        """
        batch_size, input_dim = data.shape

        # Handle default case - optimize all columns if none specified
        if columns_to_optimize is None or len(columns_to_optimize) == 0:
            columns_to_optimize = list(range(input_dim))

        # Validate column indices
        columns_to_optimize = [c for c in columns_to_optimize if 0 <= c < input_dim]
        if not columns_to_optimize:
            return data.clone(), -self.log_prob(data).sum().detach()

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
            loss = -self.log_prob(current_data).sum()

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

        return result, iteration.best_loss
    
    def log_prob(self, data, steps=None, eps=1e-3,return_separately = False,max_vectors = 32):
        """
        Computes log probability using Jacobian determinant approximation via simplex volume ratios.

        This method approximates the log probability using a novel approach based on comparing
        volumes of simplices before and after transformation. It achieves the same accuracy as
        full_log_prob but at much lower computational cost by approximating the Jacobian
        determinant through the ratio of simplex volumes in the data and latent spaces.

        The algorithm works by:
        1. Creating a small simplex around each data point
        2. Transforming the simplex vertices through the inverse flow
        3. Computing the volume ratio between the original and transformed simplices
        4. Using this ratio as an approximation of the Jacobian determinant

        Args:
            data (torch.Tensor): Input tensor for which to compute log probability.
                               Expected shape: [batch_size, input_dim]
            steps (int, optional): Number of integration steps to use for transformation.
                                 If None, uses the model's default steps.
            eps (float): Small epsilon value for simplex perturbation (default: 1e-3)
            return_separately (bool): If True, returns prior logprob, jacobian logdet,
                                    and latent variables separately. If False, returns
                                    the combined log probability (default: False)
            max_vectors: Maximum number of vectors to use for simplex determinant approximation.

        Returns:
            torch.Tensor or tuple: If return_separately is False, returns combined log
                                 probability tensor of shape [batch_size]. If True,
                                 returns a tuple of (prior_logp, logdet_approx, X)
        """
        
        
        if not steps: steps = self.default_steps
        Y = data.to(self.device)

        # generate N-dimensional simplex
        simplex_points = generate_unit_simplex_vertices(self.in_dim).to(self.device)*eps

        # simplex that have some point at origin 0
        shifted_simplex=simplex_points[:-1,:]-simplex_points[-1]

        # log area of original simplex
        original_simplex_area_log = shifted_simplex[:max_vectors,:max_vectors].slogdet()[1]

        # make shapes match
        simplex_points = simplex_points.view(*([1]*(Y.ndim-1)),*simplex_points.shape)

        # shift Y to sphere points of simplex
        Y_neighbors = Y[...,None,:] + simplex_points[:max_vectors]  # (B, n_neighbors, ...dim)

        # Compute priors for all neighbors
        X_neighbors = self.to_prior(Y_neighbors, steps)

        # get area of transformed simplex
        transformed_simplex = X_neighbors[...,:-1,:]-X_neighbors[...,[-1],:]
        transformed_simplex_area_log = transformed_simplex[...,:max_vectors,:max_vectors].slogdet()[1]

        # area ratio is our jacobian determinant approximation
        logdet_approx = transformed_simplex_area_log - original_simplex_area_log + self.in_dim*math.log(self.in_dim)

        X = self.to_prior(Y,steps)
        prior_logp = Normal(0,1).log_prob(X).sum(-1)

        if return_separately:
            return prior_logp,logdet_approx,X
        # this stuff perfectly match log prob structure
        return prior_logp+logdet_approx
        