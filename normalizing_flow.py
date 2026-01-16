from torch.distributions import Normal
from kemsekov_torch.residual import Residual
from kemsekov_torch.common_modules import mmd_rbf
from typing import Callable, Generator, Literal, Optional
from copy import deepcopy
import torch
import torch.nn as nn
from invertible_nn import *

class NormalizingFlow:
    """
    Wrapper around your InvertibleSequential + flow_nll_loss training loop.

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
        *,
        input_dim: int,
        hidden_dim: int = 32,
        layers: int = 3,
        device: Optional[str] = 'cpu',
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self._build_model().to(self.device)
        self.best_trained_model = None

    def to(self,device):
        self.device=device
        self.model=self.model.to(device)
        if self.best_trained_model:
            self.best_trained_model=self.best_trained_model.to(device)

    def _build_model(self) -> nn.Module:
        if self.input_dim % 2 != 0:
            raise ValueError(
                f"input_dim must be even for InvertibleScaleAndTranslate(input.chunk(2)). Got {self.input_dim}."
            )

        norm = nn.RMSNorm
        act = nn.ReLU
        dropout = lambda: nn.Dropout(p=0.05)
        # act = nn.SiLU
        
        half = self.input_dim // 2
        blocks = []
        for i in range(self.layers):
            steps = [
                nn.Linear(half, self.hidden_dim),
                
                Residual([
                    norm(self.hidden_dim),
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ],init_at_zero=True),
                Residual([
                    norm(self.hidden_dim),
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ],init_at_zero=True),
                dropout(),
                
                norm(self.hidden_dim),
                act(),

                nn.Linear(self.hidden_dim, self.input_dim),
            ]
            if i==self.layers-1 and "Norm" in str(steps[-1]):
                steps=steps[:-1]
            blocks.append(
                InvertibleScaleAndTranslate(
                    model=nn.Sequential(*steps),
                    dimension_split=-1,
                    non_linearity=InvertibleIdentity
                    # non_linearity=InvertibleTanh
                    # non_linearity=SmoothSymmetricSqrt
                    # non_linearity=InvertibleLeakyReLU
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
        return (self.best_trained_model or self.model).inverse(torch.randn((count,self.input_dim)))
    def log_prob(self, data : torch.Tensor) -> torch.Tensor:
        model = self.best_trained_model or self.model
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
        m = (self.best_trained_model or self.model)
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
        data_renoise=0.03,
        lr: float = 1e-2,
        grad_clip_max_norm: Optional[float] = 1,
        debug: bool = False,
        scheduler : Literal['exponential','cosine'] = 'cosine'
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
        if data.ndim != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Expected data shape [N, {self.input_dim}], got {tuple(data.shape)}")

        batch_size = min(batch_size,data.shape[0])
        data = data.to(self.device)
        
        data_renoise *= data.std(0).median()

        self.model.train()
        
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr,weight_decay=1e-4)
        
        self.best_loss = float("inf")
        self.best_trained_model = deepcopy(self.model).to(self.device)
        self.improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        total_steps = len(slices)*epochs
        
        if scheduler=='cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,total_steps)
        if scheduler=='exponential':
            sch = torch.optim.lr_scheduler.ExponentialLR(optim,(0.15)**(1/total_steps))
        
        try:
            for epoch in range(epochs):
                if debug and self.improved:
                    print(f"Epoch {epoch}: best_loss={self.best_loss:0.3f}")
                self.improved = False

                # shuffle each epoch
                perm = torch.randperm(n, device=self.device)
                data_shuf = data[perm]

                for start in slices:
                    batch = data_shuf[start : start + batch_size]
                    
                    if data_renoise>0:
                        batch=batch+torch.randn_like(batch)*data_renoise
                    
                    optim.zero_grad(set_to_none=True)  # set_to_none saves mem and can be faster [web:399]
                    loss = flow_nll_loss(self.model, batch, sum_dim=-1).mean()

                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_trained_model = deepcopy(self.model).to(self.device)
                        self.improved = True
                    loss.backward()
                    
                    if grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm,
                            norm_type=2.0,
                        )

                    optim.step()
                    sch.step()
                    
        except KeyboardInterrupt:
            if debug: print("Stop training")
        if debug and self.improved:
            print(f"Last Epoch {epoch}: best_loss={self.best_loss:0.3f}")
        self.model.eval()
        return self.best_trained_model.eval()
    
    def to_prior(self,data : torch.Tensor) -> torch.Tensor:
        """
        Converts data tensor to latent space (standard normal dist)
        """
        return (self.best_trained_model or self.model)(data)[0]
    
    def to_target(self,latent_prior : torch.Tensor) -> torch.Tensor:
        """
        Converts data tensor to target posterior space(dataset distribution)
        """
        return (self.best_trained_model or self.model).inverse(latent_prior)

    def conditional_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        num_samples: int,
        noise_scale: float = 0.01,
        steps: int = 2,
        lr: float = 1,
    ) -> torch.Tensor:
        """
        Sample from p(X | X[c_i] = v_i) using constrained latent space optimization with Langevin dynamics.

        Args:
            constraint: Constraint loss function. Accepts generated target in (num_samples,dim) shape and returns loss (scalar tensor) that defines condition for sampling.
            num_samples: Number of samples to generate
            noise_scale: Scale of noise added during Langevin dynamics (default 0.01)
            steps: Number of optimization steps (default 2)
            lr: Learning rate for the optimization (default 1)

        Returns:
            torch.Tensor: Samples of shape [num_samples, input_dim] satisfying the conditions
        """
        if self.best_trained_model is None:
            raise ValueError("Model must be trained before conditional sampling. Call fit() first.")

        model = self.best_trained_model or self.model
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
            L_prior = ((z * z).mean()-original_prior)**2

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