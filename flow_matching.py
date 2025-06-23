import torch

def nmse_loss(preds, targets, eps=1e-8):
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    return (((preds - targets) ** 2).mean(-1) / (targets ** 2).mean(-1).clamp(min=eps)).mean()

def direction_consistency_loss(v, x0, x1, eps=1e-8):
    """
    Computes direction consistency loss between predicted velocity v and 
    ground-truth direction from x0 to x1.
    
    Args:
        v (Tensor): predicted velocity, shape [B, C, H, W]
        x0 (Tensor): start points, shape [B, C, H, W]
        x1 (Tensor): end points, shape [B, C, H, W]
        eps (float): small number to avoid division by zero in normalization

    Returns:
        Tensor: weighted direction consistency loss (scalar)
    """
    target_dir = x1 - x0
    v_norm = F.normalize(v.flatten(1), dim=1, eps=eps)
    target_norm = F.normalize(target_dir.flatten(1), dim=1, eps=eps)
    dot_product = (v_norm * target_norm).sum(-1)
    loss = 1-dot_product.mean()
    return loss

def flow_matching_pair(model,input_domain,target_domain):
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
    
    Returns:
        Tuple[Tensor,Tensor]:
        1. Predicted direction
        2. Ground truth direction
        3. Normalized MSE loss
    """
    # generate time in range [0;1]
    time = torch.rand(input_domain.shape[0],device=input_domain.device)
    target = target_domain-input_domain
    
    time_expand = time[:,*([None]*len(target_domain.shape[1:]))]
    xt = (1-time_expand)*input_domain+time_expand*target_domain
    
    pred_direction = model(xt,time)
    
    return pred_direction,target, nmse_loss(pred_direction,target)

def sample_with_euler_integrator(model, x0, steps, churn_scale=0.005, inverse=False):
    """
    Samples from a flow-matching model with Euler integration.

    Args:
        model: Callable vθ(x, t) predicting vector field/motion.
        x0: Starting point (image or noise tensor).
        steps: Number of Euler steps.
        churn_scale: Amount of noise added for stability each step.
        inverse (bool): If False, integrate forward from x0 to x1 (image → noise).
                        If True, reverse for noise → image.
    Returns:
        xt: Final sample tensor.
    """
    device = next(model.parameters()).device
    if inverse: 
        ts = torch.linspace(1, 0, steps+1, device=device)
    else:
        ts = torch.linspace(0, 1, steps+1, device=device)
    # ts=ts.sqrt()
    x0 = x0.to(device)
    xt = x0
    dt = -1/steps if inverse else 1/steps
    
    with torch.no_grad():
        for i in range(0,steps-1):
            t = ts[i]
            # optional churn noise
            noise = xt.std() * torch.randn_like(xt) + xt.mean()
            xt = churn_scale * noise + (1 - churn_scale) * xt

            pred = model(xt, t[None])  # ensure shape (1,) or (batch,)
            # forward or reverse Euler update
            xt = xt + dt * pred

    return xt
