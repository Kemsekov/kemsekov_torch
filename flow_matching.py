from typing import Callable
import torch
import torch.nn.functional as F

class FlowMatching:
    def __init__(self,time_scaler : Callable[[torch.Tensor],torch.Tensor] = None,eps=1e-2):
        self.eps = eps
        if time_scaler is None:
            time_scaler=lambda x:1
        self.time_scaler=time_scaler
    
    def flow_matching_pair(self,model,input_domain,target_domain):
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
            3. Time
        """
        # generate time in range [0;1]
        time = torch.rand(input_domain.shape[0],device=input_domain.device)
        
        time_expand = time[:,*([None]*(target_domain.dim()-1))]
        xt = (1-time_expand)*input_domain+time_expand*target_domain
        
        pred_direction = model(xt,time)
        
        #original
        target = (target_domain-input_domain)*self.time_scaler(time_expand)
        
        return pred_direction,target, time

    def sample(self,model, x0, steps, churn_scale=0.0, inverse=False,return_intermediates = False):
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
        try:
            device = next(model.parameters()).device
        except:
            device = 'cpu'
        if inverse: 
            ts = torch.linspace(1, 0, steps+1, device=device)
        else:
            ts = torch.linspace(0, 1, steps+1, device=device)

        x0 = x0.to(device)
        xt = x0
        dt = -1/steps if inverse else 1/steps
        
        intermediates = []
        with torch.no_grad():
            for i in range(0,steps):
                t = ts[i]
                
                # optional churn noise
                if churn_scale>0:
                    noise = xt.std() * torch.randn_like(xt) + xt.mean()
                    xt = churn_scale * noise + (1 - churn_scale) * xt
                
                t_expand = t.expand(x0.shape[0])
                
                t_scaler = self.time_scaler(t)+self.eps
                # original
                pred = model(xt, t_expand)/t_scaler
                
                # forward or reverse Euler update
                xt = xt + dt * pred
                if return_intermediates:
                    intermediates.append(xt)
        
        if return_intermediates:
            return xt, intermediates
        return xt
