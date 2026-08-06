"""
Integration samplers for flow-matching models.

sample_base: Sobol-based low-discrepancy Gaussian prior sampling.
euler / momentum_heun / heun / rk3 / rk2 / one_step: numerical
integration of the learned velocity field.
"""
import math
import torch
import torch.nn as nn
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

def momentum_heun(model, x0, steps, churn_scale=0.0, inverse=False, return_intermediates=False, time_transform : nn.Module = nn.Identity(),no_grad_model = False):
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
    
    # total_evals=0
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
            prev_pred = pred(xt, t_expand_current)

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
        # total_evals+=1
        
        if return_intermediates:
            intermediates.append(xt.clone())
    
    if return_intermediates:
        return xt, intermediates
    # print("total_evals",total_evals)
    return xt

def heun(model, x0, steps, churn_scale=0.0, inverse=False, return_intermediates=False, time_transform: nn.Module = nn.Identity(), no_grad_model=False):
    device = x0.device
    if inverse:
        ts = torch.linspace(1, 0, steps+1, device=device)
    else:
        ts = torch.linspace(0, 1, steps+1, device=device)

    ts = time_transform(ts[:, None])[:, 0]
    dt = ts[1:] - ts[:-1]
    xt = x0.to(device)
    intermediates = []
    
    pred_fn = (lambda x, t: model(x, t)) if not no_grad_model else (lambda x, t: model(x, t).detach())
    
    for i in range(steps):
        t_current = ts[i]
        t_next = ts[i+1]
        
        if churn_scale > 0:
            noise = xt.std() * torch.randn_like(xt) + xt.mean()
            xt = churn_scale * noise + (1 - churn_scale) * xt
        
        t_expand_current = t_current.expand(x0.shape[0])
        t_expand_next = t_next.expand(x0.shape[0])
        
        # 1-е вычисление: производная в текущей исправленной точке
        pred_current = pred_fn(xt, t_expand_current)
        
        # Предиктор (Эйлер)
        x_pred = xt + dt[i] * pred_current
        
        # 2-е вычисление: производная в прогнозной точке
        pred_next = pred_fn(x_pred, t_expand_next)
        
        # Корректор (Хёйн)
        xt = xt + dt[i] * 0.5 * (pred_current + pred_next)
        
        if return_intermediates:
            intermediates.append(xt.clone())
            
    return (xt, intermediates) if return_intermediates else xt

def rk3(model, x0, churn_scale=0.0, inverse=False, return_intermediates=False, left = 0.0, right = 1.0):
    device = x0.device
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
    device = x0.device
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