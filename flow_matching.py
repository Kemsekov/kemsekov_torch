from copy import deepcopy
from typing import Callable, Optional
from kemsekov_torch.common_modules import Residual
from kemsekov_torch.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        device = list(model.parameters())[0].device
        if inverse: 
            ts = torch.linspace(1, 0, steps+1, device=device)
        else:
            ts = torch.linspace(0, 1, steps+1, device=device)

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

class FlowModel1d(nn.Module):
    """
    Flow-matching model for 1-dimensional (vector) data
    """
    def __init__(self, in_dim,hidden_dim=32,residual_blocks=3,dropout_p=0.05) -> None:
        super().__init__()
        self.in_dim=in_dim
        norm = nn.RMSNorm
        act = nn.GELU
        self.time_emb = nn.Sequential(
            nn.Linear(1,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
        )
        self.expand = nn.Linear(in_dim,hidden_dim)
        
        self.residual_blocks = nn.ModuleList([
            nn.ModuleList([
                Residual([
                    norm(hidden_dim),
                    act(),
                    nn.Linear(hidden_dim,hidden_dim),
                ]),
                nn.Sequential(
                    nn.Linear(hidden_dim,hidden_dim),
                    nn.Sigmoid()
                )
            ]) for i in range(residual_blocks)
        ])
        
        self.dropout = nn.Dropout(dropout_p)
        self.collapse = nn.Linear(hidden_dim,in_dim)
        self.default_steps=24
        
    def forward(self,x : torch.Tensor,t : torch.Tensor):
        if t.ndim==1: t=t[:,None]
        x = self.expand(x)+self.time_emb(t)
        
        for m,temb in self.residual_blocks:
            x = m(x*temb(x))
        
        x = self.dropout(x)
        return self.collapse(x)
        
    def fit(
        model,
        data: torch.Tensor,
        prior_dataset : Optional[torch.Tensor] = None,
        batch_size: int = 512,
        epochs: int = 30,
        contrastive_loss_weight=0.1,
        lr: float = 0.025,
        grad_clip_max_norm: Optional[float] = 1,
        debug: bool = False,
        reflow_window = 1,
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
        

        device = list(model.parameters())[0].device

        batch_size = min(batch_size,data.shape[0])
        data = data.to(device)
        
        if prior_dataset is not None:
            prior_dataset=prior_dataset.to(device)
        
        model.train()
        fm = FlowMatching()
        optim = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4)
        
        best_loss = float("inf")
        best_r2 = -1e8
        best_trained_model = deepcopy(model).to(device)
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        total_steps = len(slices)*epochs
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,total_steps)
        lossf = torch.nn.functional.mse_loss
        
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
                    sch.step()
                    
                    r2 = r2_score(pred_dir,target_dir)
                    losses.append(loss)
                    r2s.append(r2)
                    
                mean_loss = sum(losses)/len(losses)
                mean_r2 = sum(r2s)/len(r2s)
                if mean_r2 > best_r2:
                    best_loss = mean_loss.item()
                    best_trained_model = deepcopy(model).to(device)
                    best_r2 = mean_r2
                    improved = True
        except KeyboardInterrupt:
            if debug: print("Stop training")
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}")
        
        # update current model with best checkpoint
        with torch.no_grad():
            for p1,p2 in zip(model.parameters(),best_trained_model.parameters()):
                p1+=p2-p1

    def reflow(self,
               dataset_size,
               batch_size,
               epochs_per_window_step=10,
               window_steps=5,
               lr=1e-2,
               debug = False
        ):
        current_model = deepcopy(self).train()
        for w in torch.linspace(1/window_steps,1,window_steps):
            if debug:
                print(f"Training on t=[0.00:{w:0.2f}]")
            prior = torch.randn((dataset_size,self.in_dim))
            with torch.no_grad():
                data = current_model.to_target(prior)
            self.fit(
                data,
                prior_dataset=prior,
                debug=debug,
                batch_size=batch_size,
                epochs=epochs_per_window_step,
                lr=lr,
                reflow_window=w,
            )
        self.default_steps=5
        self.eval()
    
    def to_prior(self,data : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        fm = FlowMatching()
        return fm.sample(self,data,steps,inverse=True)
    
    def to_target(self,normal_noise : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        fm = FlowMatching()
        return fm.sample(self,normal_noise,steps)
    
    def sample(self,count,steps=None):
        if not steps: steps = self.default_steps
        return self.to_target(torch.randn((count,self.in_dim)),steps)
        