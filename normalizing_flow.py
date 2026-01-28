from torch.distributions import Normal
from kemsekov_torch.residual import Residual
from kemsekov_torch.common_modules import mmd_rbf,Prod, AddConst
from typing import Callable, Generator, Literal, Optional
from copy import deepcopy
import torch
import torch.nn as nn
from invertible_nn import *

class LossNormalizer1d(nn.Module):
    def __init__(self, in_dim,hidden_dim=32) -> None:
        super().__init__()
        self.expand = nn.Linear(in_dim,hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self,x : torch.Tensor):
        """
        Forward pass of the loss normalizer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_dim]
        Returns:
            torch.Tensor: Normalized loss weights of shape [batch_size, in_dim]
        """
        x = self.expand(x)
        return self.net(x)[:,0]

class NormalizingFlowScaler:
    """
    Data scaler for normalizing flow
    """
    def __init__(self) -> None:
        self.mean = 0
        self.std = 1
        
    def inverse(self,data):
        input_shape = list(data.shape)
        input_shape[-1]//=2
        last_dim = input_shape[-1]
        data = data.flatten(-1)[:,:last_dim]
        return (data*self.std[:,:last_dim]+self.mean[:,:last_dim]).reshape(input_shape)
        
    def transform(self,data : torch.Tensor):
        input_shape = list(data.shape)
        input_shape[-1]*=2
        data = torch.concat([data,data.log_softmax(-1)],-1).flatten(-1)
        data = (data-self.mean)/self.std
        return data.reshape(input_shape)
    
    def fit_transform(self,data : torch.Tensor):
        input_shape = list(data.shape)
        input_shape[-1]*=2
        data = torch.concat([data,data.log_softmax(-1)],-1).flatten(-1)
        self.mean = data.mean(0,keepdim=True)
        self.std = data.std(0,keepdim=True)+1e-6
        
        data = (data-self.mean)/self.std
        return data.reshape(input_shape)
        
        
class NormalizingFlow:
    """
    Wrapper around your InvertibleSequential + flow_nll_loss training loop.
    
    You must use this class alongside `NormalizingFlowScaler`

    Key features:
    - Model definition is fully determined in __init__ (input_dim is required, not inferred from data).
    - fit(...) trains on a tensor dataset and returns the best model (CPU, eval).
    - Works with flow_nll_loss that returns either:
        * loss
        * (loss, diagnostics_dict)
      (avoids "iteration over a 0-d tensor" unpacking error).
    - Optional gradient clipping via torch.nn.utils.clip_grad_norm_. [web:381]
    - Uses optimizer.zero_grad(set_to_none=True) for performance/memory. [web:399]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        layers: int = 3,
        dropout=0.05,
        device: Optional[str] = 'cpu',
        non_linearity : Union[SmoothSymmetricSqrt,InvertibleIdentity] = InvertibleIdentity,
    ):
        self.non_linearity=non_linearity
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self._build_model(dropout).to(self.device)
        self.best_trained_model = None
        self._data_mean = 0
        self._data_std = 1

    def to(self,device):
        self.device=device
        self.model=self.model.to(device)
    
    def _build_model(self,dropout_p) -> nn.Module:
        if self.input_dim % 2 != 0:
            raise ValueError(
                f"input_dim must be even for InvertibleScaleAndTranslate(input.chunk(2)). Got {self.input_dim}."
            )

        norm = nn.RMSNorm
        # norm = nn.BatchNorm1d
        # act = nn.ReLU
        act = nn.GELU
        # act = nn.SiLU
        dropout = lambda: nn.Dropout(p=dropout_p)
        input_dim = self.input_dim
        half = input_dim // 2
        blocks = []
        for i in range(self.layers):
            steps = [
                nn.Linear(half, self.hidden_dim),
                norm(self.hidden_dim),
                # dropout(),
                Residual([
                    # norm(self.hidden_dim),
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ],init_at_zero=True),
                # Residual([
                #     # norm(self.hidden_dim),
                #     act(),
                #     # dropout(),
                #     nn.Linear(self.hidden_dim, self.hidden_dim),
                # ],init_at_zero=True),

                Prod(nn.Sequential(
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    norm(self.hidden_dim),
                    # SmoothSymmetricSqrt()
                    nn.Tanh(),
                    # AddConst(1.0)
                )),
                act(),
                nn.Linear(self.hidden_dim, input_dim),
            ]
            if i==self.layers-1 and "Norm" in str(steps[-1]):
                steps=steps[:-1]
            blocks.append(
                InvertibleScaleAndTranslate(
                    model=nn.Sequential(*steps),
                    dimension_split=-1,
                    non_linearity=self.non_linearity,
                )
            )
        blocks[-1].non_linearity = InvertibleIdentity()
        return InvertibleSequential(*blocks)
    
    def MMD2_with_data(self,data : torch.Tensor) -> float:
        """
        Returns MMD^2 of sampled learned latent space with given data.
        This method can be used as a metric for evaluating how good trained model is.
        """
        with torch.no_grad():
            sampled = self.sample(len(data))
            return mmd_rbf(data,sampled)[0].item()
    
    def sample(self,count : int) -> torch.Tensor:
        """
        Generates samples drawn from trained distribution
        """
        return self.model.inverse(torch.randn((count,self.input_dim)))
    
    def log_prob(self, data : torch.Tensor) -> torch.Tensor:
        model = self.model
        z, jacobians = model(data.to(self.device))
        
        # log p(z) under standard normal
        log_pz = Normal(0, 1).log_prob(z).flatten(-1).sum(dim=-1)
        
        # log |det J|
        log_det = 0.0
        for jd in jacobians:
            log_abs_jd = torch.log(torch.abs(jd) + 1e-8)
            log_det += log_abs_jd.flatten(-1).sum(dim=-1)
        
        # log p(x) = log p(z) + log |det J|
        log_px = log_pz + log_det
        
        return log_px.to(data.device)

    def interpolate(self,dataA : torch.Tensor,dataB : torch.Tensor, N : int):
        """
        Generate N interpolated samples between dataA and dataB via latent space linear interpolation.
        
        Args:
            dataA: Starting data point tensor
            dataB: Ending data point tensor  
            N: Number of interpolation steps to generate
        
        Yields:
            torch.Tensor: Interpolated sample at each step from dataA to dataB
        """
        m = self.model
        latentsA = m(dataA)[0]
        latentsB = m(dataB)[0]
        time = torch.linspace(0,1,N)
        for i in range(N):
            t = time[i]
            interpolated = (1-t)*latentsA+t*latentsB
            yield m.inverse(interpolated)

    def optimize(self, data: torch.Tensor, lr: float = 1.0, epochs: int = 1, 
             columns_to_optimize: list[int] = None):
        """
        Optimize only specific columns of data to maximize log probability.
        
        Args:
            data: Input tensor of shape [batch_size, input_dim]
            columns_to_optimize: List of column indices to optimize (0-based). 
                                If None or empty, all columns will be optimized.
            
        Returns:
            Optimized data tensor, final loss
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
            optimizer.zero_grad()
            
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

    def fit(
        self,
        data: torch.Tensor,
        batch_size: int = 512,
        epochs: int = 30,
        data_renoise_start=0.1,
        data_renoise_end=0.01,
        lr: float = 1e-2,
        grad_clip_max_norm: Optional[float] = 1,
        debug: bool = False,
        loss_normalizer_weight = 0.1,
        data_prior : Optional[torch.Tensor] = None,
        scheduler : Literal['exponential','cosine'] = 'cosine',
    ) -> nn.Module:
        """
        Train on `data` and return best model.

        Args:
            data: Tensor of shape [N, input_dim].
            batch_size: Batch size.
            epochs: Epoch count.
            data_renoise_start: dataset renoise factor. How much renoise training data at the first epochs.
            data_renoise_end: lowest dataset renoise factor.
            lr: AdamW learning rate.
            grad_clip_max_norm: If not None, clip global grad norm to this value. [web:381]
            debug: If True, prints when best loss improves.

        Returns:
            trained_model: Best model on CPU in eval() mode.
        """
        if data.ndim != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Expected data shape [N, {self.input_dim}], got {tuple(data.shape)}")

        batch_size = min(batch_size,data.shape[0])
        data = data.to(self.device)

        if data_prior is not None:
            data_prior = data_prior.to(self.device)
        
        data_renoise_start *= data.std(0).median()
        data_renoise_end *= data.std(0).median()

        self.model.train()
        loss_normalizer = LossNormalizer1d(self.input_dim,hidden_dim=self.hidden_dim)
        
        optim = torch.optim.AdamW(list(self.model.parameters())+list(loss_normalizer.parameters()), lr=lr,fused=True)
        
        best_loss = float("inf")
        best_trained_model = deepcopy(self.model).to(self.device)
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        total_steps = len(slices)*epochs
        
        if scheduler=='cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,total_steps)
        if scheduler=='exponential':
            sch = torch.optim.lr_scheduler.ExponentialLR(optim,(0.15)**(1/total_steps))
        try:
            for epoch in range(epochs):
                if debug and improved:
                    print(f"Epoch {(str(epoch)+"   ")[:3]}: best_loss={str(best_loss)[:5]} renoise_level={str(renoise_level.item())[:5]}")
                improved = False

                # shuffle each epoch
                perm = torch.randperm(n, device=self.device)
                data_shuf = data[perm]
                if data_prior is not None:
                    prior_shuf = data_prior[perm]

                losses = []
                part = (epoch+1)/epochs
                renoise_level = (data_renoise_start*(1-part)+data_renoise_end*part)
                for start in slices:
                    batch = data_shuf[start : start + batch_size]
                    
                    if renoise_level>0:
                        batch=batch+torch.randn_like(batch)*renoise_level
                    
                    optim.zero_grad(set_to_none=True)
                    z,jac = self.model(batch)
                    loss_weight = loss_normalizer(z) # log(1/nil)=-log(nil)
                    
                    nil,log_det = flow_nll_loss(z,jac, batch, sum_dim=[-1])
                    
                    nil+=8
                    
                    model_loss = (loss_weight.detach().exp()*nil).mean()
                    
                    with torch.no_grad():
                        expected_loss = -nil.clamp(1e-7).log().detach()
                        
                    normalizer_loss = torch.nn.functional.mse_loss(expected_loss,loss_weight)
                    loss = model_loss+normalizer_loss*loss_normalizer_weight
                    if data_prior is not None:
                        prior_batch = prior_shuf[start : start + batch_size]
                        loss+=(z-prior_batch).pow(2).mean()

                    loss.backward()
                    
                    if grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm,
                            norm_type=2.0,
                        )

                    optim.step()
                    sch.step()
                    losses.append(nil.mean())
                mean_loss = sum(losses)/len(losses)
                if mean_loss < best_loss:
                    best_loss = mean_loss.item()
                    best_trained_model = deepcopy(self.model)
                    improved = True
        except KeyboardInterrupt:
            if debug: print("Stop training")
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}")
        self.model.eval()
        with torch.no_grad():
            for a,b in zip(self.model.parameters(),best_trained_model.parameters()):
                a.copy_(b.to(a.device))
        
    
    def to_prior(self,data : torch.Tensor) -> torch.Tensor:
        """
        Converts data tensor to latent space (standard normal dist)
        """
        return self.model(data)[0]
    
    def to_target(self,latent_prior : torch.Tensor) -> torch.Tensor:
        """
        Converts data tensor to target posterior space(dataset distribution)
        """
        return self.model.inverse(latent_prior)

    def conditional_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        num_samples: int,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 1.0
    ) -> torch.Tensor:
        """
        Sample from p(X | X[c_i] = v_i) using constrained latent space optimization with Langevin dynamics.

        Args:
            constraint: Constraint loss function. Accepts generated target in (num_samples,dim) shape and returns loss (scalar tensor) that defines condition for sampling.
            num_samples: Number of samples to generate
            noise_scale: Scale of noise added during Langevin dynamics (default 0.00). Increasing this value will result in samples more spread from condition. Values around [0 to 0.05] are generally good enough.
            steps: Number of optimization steps (default 2)
            lr: Learning rate for the optimization (default 1)
            mode_closeness_weight: Weight for trying to sample closer to distribution mode. Increasing this value make samples cluster more around closest distribution mode, potentially leading to mode collapse (all samples are the same). Values [0 to 2] are generally good enough.

        Returns:
            torch.Tensor: Samples of shape [num_samples, input_dim] satisfying the conditions
        """

        model = self.model
        model.eval()

        # Initialize z from standard normal distribution
        z = torch.randn(num_samples, self.input_dim, device=self.device, requires_grad=True)

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
            x = model.inverse(z)

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
            final_x = model.inverse(self._iteration.best_sample)

        return final_x