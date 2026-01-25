from copy import deepcopy
import math
from typing import Callable, Optional
from kemsekov_torch.common_modules import Residual
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.common_modules import mmd_rbf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def euler(model, x0, steps, churn_scale=0.0, inverse=False,return_intermediates = False):
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
    x0 = x0.to(device)
    xt = x0
    dt = -1/steps if inverse else 1/steps
    
    intermediates = []
    
    for i in range(0,steps):
        t = ts[i]
        
        # optional churn noise
        if churn_scale>0:
            noise = xt.std() * torch.randn_like(xt) + xt.mean()
            xt = churn_scale * noise + (1 - churn_scale) * xt
        t_expand = t.expand(x0.shape[0])
        
        pred = model(xt, t_expand)
        
        # forward or reverse Euler update
        xt = xt + dt * pred
        if return_intermediates:
            intermediates.append(xt)
    
    if return_intermediates:
        return xt, intermediates
    return xt

def heun(model, x0, steps, churn_scale=0.0, inverse=False, return_intermediates=False,base_shift=0.0):
    device = list(model.parameters())[0].device
    shift = base_shift/steps
    if inverse:
        ts = torch.linspace(1-shift, shift, steps+1, device=device)  # steps intervals = steps+1 points
        dt = ts[1]-ts[0]
    else:
        ts = torch.linspace(shift, 1-shift, steps+1, device=device)
        dt = ts[1]-ts[0]
    
    x0 = x0.to(device)
    xt = x0
    intermediates = []
    
    # Store previous derivative for multi-step method
    prev_pred = None
    
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
            pred_current = model(xt, t_expand_current)
            
            # Predictor step (Euler)
            x_pred = xt + dt * pred_current
            
            # Evaluate at predicted point
            t_expand_next = t_next.expand(x0.shape[0])
            pred_next = model(x_pred, t_expand_next)
            
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
            pred_next = model(x_pred, t_expand_next)
            
            # Corrector step using stored previous derivative
            xt = xt + dt * 0.5 * (prev_pred + pred_next)
            
            # Update stored derivative for next step
            prev_pred = pred_next
        
        if return_intermediates:
            intermediates.append(xt.clone())
    
    if return_intermediates:
        return xt, intermediates
    return xt

def rk3(model, x0, churn_scale=0.0, inverse=False, return_intermediates=False,interval_shift=0.):
    device = next(model.parameters()).device
    x0 = x0.to(device)
    xt = x0.clone()
    
    # === THEORETICAL DEFAULTS: Classical RK3 Butcher tableau ===
    # Forward: t ∈ [0, 1], Reverse: t ∈ [1, 0] (proper time mapping)
    # but i expect these values to be altered for flow matching model towards
    # the center by some amount
    if inverse:
        t_start =   1.0-interval_shift
        t_end =     0.0+interval_shift
    else:
        t_start =   0.0+interval_shift
        t_end =     1.0-interval_shift
    
    
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

def rk2(model, x0, churn_scale=0.0, inverse=False, return_intermediates=False,interval_shift=0.):
    device = next(model.parameters()).device
    x0 = x0.to(device)
    xt = x0.clone()
    
    # === THEORETICAL DEFAULTS: Ralston's method (optimal 2-stage RK) ===
    if inverse:
        t_start =   1.0-interval_shift
        t_end =     0.0+interval_shift
    else:
        t_start =   0.0+interval_shift
        t_end =     1.0-interval_shift
    dt = t_end - t_start
    
    intermediates = []
    
    if churn_scale > 0:
        noise = xt.std() * torch.randn_like(xt) + xt.mean()
        xt = churn_scale * noise + (1 - churn_scale) * xt
    
    # === First evaluation (k1 at start) - weight = 1/4 ===
    t_expand_start = torch.tensor([t_start], device=device).expand(x0.shape[0])
    k1 = model(xt, t_expand_start)
    
    # === Second evaluation (k2 at 2/3 point) - Ralston's optimal point ===
    t_ralston = t_start + (2/3) * dt  # = 1/3 for reverse, 2/3 for forward
    x_ralston = xt + (2/3) * dt * k1
    t_expand_ralston = torch.tensor([t_ralston], device=device).expand(x0.shape[0])
    k2 = model(x_ralston, t_expand_ralston)
    
    # === Ralston's weights - proven minimum error bound ===
    weight_k1 = 1/4
    weight_k2 = 3/4
    
    xt_next = xt + dt * (weight_k1 * k1 + weight_k2 * k2)
    
    if return_intermediates:
        intermediates.extend([x_ralston, xt_next])
    
    return (xt_next, intermediates) if return_intermediates else xt_next

class FlowMatching:
    def __init__(self,time_scaler : Callable[[torch.Tensor],torch.Tensor] = None,eps=1e-2):
        """
        Initializes a Flow Matching trainer and sampler.

        Parameters
        ----------
        time_scaler : Callable[[torch.Tensor], torch.Tensor], optional
            A function that modulates the ground-truth flow direction as a function of time `t`.
            It takes a time tensor `t` (shape `[B]` or broadcastable) and returns a scaling factor
            of the same shape. This scaling is applied to the raw direction vector
            `(target_domain - input_domain)` during training.

            By default (`time_scaler=None`), the identity scaling is used: `time_scaler(t) = 1`,
            which corresponds to standard **straight-path flow matching** (i.e., linear interpolation
            with constant velocity).

            Common alternatives include:
              - `lambda t: t`: induces time-dependent velocity scaling (e.g., mimicking ODEs from diffusion).
              - `lambda t: torch.sqrt(t)` or other smooth functions to align with specific generative priors.

            The choice of `time_scaler` defines the **vector field** that the model learns to approximate.
            It must be consistent between training and sampling: during sampling, the model output is
            divided by `time_scaler(t) + eps` to invert this scaling and recover the true trajectory.

        eps : float, optional (default: 1e-2)
            Small positive constant added to the time scaler output during sampling to avoid division
            by zero when `time_scaler(t)` approaches zero (e.g., at `t=0` if `time_scaler(t) = t`).
            This ensures numerical stability in the Euler integration step.

        Notes
        -----
        - During **training** (`flow_matching_pair` or `contrastive_flow_matching_pair`):
            The target direction is computed as:
                `target = (target_domain - input_domain) * time_scaler(t)`

        - During **sampling** (`sample` method):
            The model's raw output (which approximates the scaled vector field) is corrected by:
                `pred = model(xt, t) / (time_scaler(t) + eps)`
            This recovers the unscaled velocity for integration, assuming the model learned
            `v(x_t, t) ≈ (target - input) * time_scaler(t)`.

        - **Consistency is crucial**: The same `time_scaler` must be used in both training and inference.
          Mismatched scalers will lead to incorrect trajectories and poor sample quality.

        Example
        -------
        To replicate standard straight-path flow matching (constant velocity):
            fm = FlowMatching()  # uses time_scaler = lambda t: 1

        To use a diffusion-like scaling (velocity vanishes at t=0):
            fm = FlowMatching(time_scaler=lambda t: t)
        """
        
        self.eps = eps
        if time_scaler is None:
            time_scaler=lambda x:1
        self.time_scaler=time_scaler
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
        
        time_expand = time[:,*([None]*(target_domain.dim()-1))]
        xt = (1-time_expand)*input_domain+time_expand*target_domain
        
        pred_direction = model(xt,time)
        
        #original
        target = (target_domain-input_domain)*self.time_scaler(time_expand)
        
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
        bsz = input_domain.shape[0]
        time_expand = time[:, *([None] * (target_domain.dim() - 1))]
        xt = (1 - time_expand) * input_domain + time_expand * target_domain

        pred_direction = model(xt, time)

        target = (target_domain - input_domain) * self.time_scaler(time_expand)

        # Prepare negative samples by shuffling the batch
        idx = torch.randperm(bsz, device=input_domain.device)
        input_neg = input_domain[idx]
        target_neg = target_domain[idx]
        time_expand_neg = time_expand[idx]

        target_neg_vec = (target_neg - input_neg) * self.time_scaler(time_expand_neg)

        return pred_direction, target, target_neg_vec, time_expand
    def integrate(self,model, x0, steps, churn_scale=0.0, inverse=False, return_intermediates=False):
        match steps:
            case 1: return euler(model,x0,steps,churn_scale,inverse,return_intermediates)
            case 2: return rk2(model,x0,churn_scale,inverse,return_intermediates)
            case 3: return rk3(model,x0,churn_scale,inverse,return_intermediates)
            case _: return heun(model,x0,steps-1,churn_scale,inverse,return_intermediates)

from torch.func import vmap, jacrev

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

class FlowModel1d(nn.Module):
    """
    Flow-matching model for 1-dimensional (vector) data
    """
    def __init__(self, in_dim,hidden_dim=32,residual_blocks=3,dropout_p=0.0,device='cpu') -> None:
        super().__init__()
        self.in_dim=in_dim
        norm = nn.RMSNorm
        act = nn.GELU
        self.time_emb = nn.Sequential(
            nn.Linear(1,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),
        )
        self.expand = nn.Linear(in_dim,hidden_dim)
        
        # self.dropout = nn.Dropout(dropout_p)
        
        self.residual_blocks = nn.ModuleList([
            nn.ModuleList([
                Residual([
                    norm(hidden_dim),
                    act(),
                    nn.Linear(hidden_dim,hidden_dim),
                ]),
                nn.Sequential(
                    nn.Linear(hidden_dim,hidden_dim),
                    nn.Sigmoid(),
                )
            ]) for i in range(residual_blocks)
        ])
        
        self.collapse = nn.Linear(hidden_dim,in_dim)
        self.default_steps=12
        self.to(device)
        self.eval()
        
    def forward(self,x : torch.Tensor,t : torch.Tensor):
        if t.ndim==1: t=t[:,None]
        time = self.time_emb(t)
        while time.ndim<x.ndim:
            time = time[:,None]

        expand = self.expand(x)
        x = expand+time
        for m,temb in self.residual_blocks:
            x = m(x*temb(x))
        
        return self.collapse(x)
    
    def to(self,device):
        self.device=device
        return super().to(device)
    
    def fit(
        model,
        data: torch.Tensor,
        batch_size: int = 512,
        epochs: int = 100,
        contrastive_loss_weight=0.1,
        lr: float = 0.02,
        grad_clip_max_norm: Optional[float] = 1,
        debug: bool = False,
        prior_dataset : Optional[torch.Tensor] = None,
        reflow_window = 1,
        lossf = F.mse_loss,
        scheduler = True,
    ) -> nn.Module:
        """
        Train on `data` and return best model.

        Args:
            data: Tensor of shape [N, input_dim].
            batch_size: Batch size.
            epochs: Epoch count.
            data_renoise: dataset renoise factor. This is very important parameter and must be finetuned. Lays in range [0.01,0.1]
            lr: AdamW learning rate.
            grad_clip_max_norm: If not None, clip global grad norm to this value. [web:381]
            debug: If True, prints when best loss improves.

        Returns:
            trained_model: Best model on CPU in eval() mode.
        """
        device = model.device

        batch_size = min(batch_size,data.shape[0])
        data = data.to(device)
        
        if prior_dataset is not None:
            prior_dataset=prior_dataset.to(device)
        
        model.train()
        fm = FlowMatching()
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        best_r2 = -1e8
        
        best_trained_model = deepcopy(model)
        
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        total_steps = len(slices)*epochs
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,total_steps)
        try:
            for epoch in range(epochs):
                if debug and improved:
                    print(f"Epoch {epoch}: best_loss={best_loss:0.3f}\tbest r2={best_r2:0.3f}")
                improved = False
                
                # shuffle each epoch
                perm = torch.randperm(n, device=device)
                data_shuf = data[perm]
                
                if prior_dataset is not None:
                    prior_shuf = prior_dataset[perm]

                losses = []
                r2s = []
                for start in slices:
                    batch = data_shuf[start : start + batch_size]

                    optim.zero_grad(set_to_none=True)  # set_to_none saves mem and can be faster [web:399]
                    prior_batch=torch.randn_like(batch,device=device)
                    time = torch.rand(batch.shape[0],device=device)
                    if prior_dataset is not None:
                        prior_batch = prior_shuf[start : start + batch_size]
                        time*=reflow_window
                            
                    pred_dir,target_dir,contrast_dir,t = \
                        fm.contrastive_flow_matching_pair(
                            model,
                            prior_batch,
                            batch,
                            time=time
                        )
                    loss = lossf(pred_dir,target_dir)-contrastive_loss_weight*lossf(pred_dir,contrast_dir)
                    
                    
                    loss.backward()
                    
                    if grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=grad_clip_max_norm,
                            norm_type=2.0,
                        )
                    optim.step()
                    if scheduler: sch.step()
                    
                    r2 = r2_score(pred_dir,target_dir)
                    losses.append(loss)
                    r2s.append(r2)
                    
                mean_loss = sum(losses)/len(losses)
                mean_r2 = sum(r2s)/len(r2s)
                if mean_r2 > best_r2:
                    best_loss = mean_loss.item()
                    best_trained_model = deepcopy(model)
                    best_r2 = mean_r2
                    improved = True
        except KeyboardInterrupt:
            if debug: print("Stop training")
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}")
        
        # update current model with best checkpoint
        with torch.no_grad():
            for p1,p2 in zip(model.parameters(),best_trained_model.parameters()):
                p1.copy_(p2.to(device))
        model.eval()
    
    def mmd2_with_data(self,data : torch.Tensor) -> float:
        """
        Returns MMD^2 of sampled learned latent space with given data.
        This method can be used as a metric for evaluating how good trained model is.
        """
        with torch.no_grad():
            sampled = self.sample(len(data))
            return mmd_rbf(data.to(self.device),sampled)[0].item()
    
    def reflow(self,
               dataset_size=4096,
               batch_size=512,
               epochs_per_window_step=10,
               window_steps=4,
               lr=0.01,
               debug = False,
               contrastive_loss_weight=0.02
        ):
        current_model = deepcopy(self).train()
        for w in torch.linspace(1/window_steps,1,window_steps):
            if debug: print(f"Training on t=[0.00:{w:0.2f}]")
            prior = torch.randn((dataset_size,self.in_dim))
            with torch.no_grad():
                data = current_model.to_target(prior)
            self.fit(
                data=data,
                prior_dataset=prior,
                debug=debug,
                batch_size=batch_size,
                epochs=epochs_per_window_step,
                lr=lr,
                reflow_window=w,
                # lossf = F.smooth_l1_loss,
                scheduler=False,
                contrastive_loss_weight=contrastive_loss_weight
            )
        self.default_steps=4
        self.eval()
    
    def to_prior(self,data : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        input_device = data.device
        fm = FlowMatching()
        return fm.integrate(self,data.to(self.device),steps,inverse=True).to(input_device)
    
    def to_target(self,normal_noise : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        input_device = normal_noise.device
        fm = FlowMatching()
        return fm.integrate(self,normal_noise.to(self.device),steps).to(input_device)
    
    def sample(self,num_samples,steps=None):
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
    
    # TODO: make a publication about this method
    def log_prob(self, data, steps=None, eps=1e-3,return_separately = False):
        """
        Jacobian det approx via pairwise L2 distances of perturbed neighbors.
        Gives exact same result as `full_log_prob` but at **much** lower computational cost.
        
        This piece of code is just brilliant, i don't understand myself how i've come up with it,
        but this stuff perfectly computes log prob of flow matching model at very low cost.
        
        return_separately: returns prior_logprob and jacob_log separately
        """
        if not steps: steps = self.default_steps
        Y = data.to(self.device)
        
        # generate N-dimensional simplex
        simplex_points = generate_unit_simplex_vertices(self.in_dim).to(self.device)*eps
        
        # simplex that have some point at origin 0
        shifted_simplex=simplex_points[:-1,:]-simplex_points[-1]

        # log area of original simplex
        original_simplex_area_log = shifted_simplex.slogdet()[1]
        
        # make shapes match
        simplex_points = simplex_points.view(*([1]*(Y.ndim-1)),*simplex_points.shape)
        
        # shift Y to sphere points of simplex
        Y_neighbors = Y[...,None,:] + simplex_points  # (B, n_neighbors, ...dim)
        
        # Compute priors for all neighbors
        X_neighbors = self.to_prior(Y_neighbors, steps)
        
        # get area of transformed simplex
        transformed_simplex = X_neighbors[...,:-1,:]-X_neighbors[...,[-1],:]
        transformed_simplex_area_log = transformed_simplex.slogdet()[1]
        
        # area ratio is our jacobian determinant approximation
        logdet_approx = transformed_simplex_area_log - original_simplex_area_log + self.in_dim*math.log(self.in_dim)
        
        X = self.to_prior(Y,steps)
        prior_logp = Normal(0,1).log_prob(X).sum(-1)

        if return_separately:
            return prior_logp,logdet_approx
        # this stuff perfectly match log prob structure
        return prior_logp+logdet_approx
        