"""
Core flow-matching machinery: FlowMatching (training-pair generation,
integration dispatch), LossNormalizer1d, ReZero (zero_module),
FusedFlowResidual block and get_fm_optim_groups (decay grouping).
"""
import torch
import torch.nn as nn
from kemsekov_torch.common_modules import Prod, Residual
from kemsekov_torch.fm.samplers import momentum_heun, one_step, rk2, rk3


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
            dtype = self.one_weights.dtype
        else:
            device=None
            dtype=torch.float32
        with torch.no_grad():
            start_time = self.time_scaler(0.5)
            self.one_weights     = torch.nn.Parameter(torch.tensor([start_time,  0.5, 1,0,0],device=device,dtype=dtype))
            self.one_weights_inv = torch.nn.Parameter(torch.tensor([1-start_time,-0.5,1,0,0],device=device,dtype=dtype))
            self.rk2_weights     = torch.nn.Parameter(torch.tensor([start_time,   1.0,  0.5, 0.5, 1.0, 1.0, 0.0, 0.0],device=device,dtype=dtype))
            self.rk2_weights_inv = torch.nn.Parameter(torch.tensor([1-start_time, 0.0, -0.5, -0.5, -1.0, 1.0, 0.0, 0.0],device=device,dtype=dtype))
    
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

    def contrastive_flow_matching_pair(self, model, input_domain, target_domain, time=None, idx=None):
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
        # idx can be supplied externally (e.g. by captured CUDA graphs, where
        # drawing RNG inside the graph is illegal)
        if idx is None:
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
            case _: return momentum_heun(model,x0,steps-1,churn_scale,inverse,return_intermediates,time_transform=self.time_scaler,no_grad_model=no_grad_model)

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
# FusedFlowResidual is provided by the Triton-kernel implementation in
# triton_residual.py (same module structure and state_dict keys as the
# original; forward routed through fused Triton kernels on CUDA fp32,
# with an eager fallback otherwise).
from kemsekov_torch.fm.triton_residual import FusedFlowResidual




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