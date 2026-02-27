from copy import deepcopy
import math
from typing import List
import torch
import torch.nn as nn
from torch.quasirandom import SobolEngine

class Residual(torch.nn.Module):
    def __init__(self,m : torch.nn.Module | List[torch.nn.Module]):
        super().__init__()
        if isinstance(m,list) or isinstance(m,tuple):
            m = torch.nn.Sequential(*m)
        self.m = m
            
    def forward(self,x):
        out = self.m(x)
        return out+x

class Prod(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x * self.net(x)

class MarginalFlow(nn.Module):
    def __init__(self, data_dim, latent_dim=1, hid=64,layers=1):
        """
        data_dim: input dataset dimension
        latent_dim: marginalized latent-space dimension. I advice you to set it to [1] or [2] at most. Even very high-dimensional data can be perfectly fit in 1-2 dimensions with marginal flow.
        """
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hid = hid
        self.sobol = SobolEngine(latent_dim, scramble=True)
        
        # Learnable parameter C for log_sigma stabilization
        # Initialized to zeros to start with zero log_sigma (unit variance)
        self.C = nn.Parameter(torch.zeros(data_dim))
        
        # Forward model (Generator / Density Estimator)
        self.fz = self._build_model(in_dim=latent_dim, out_dim=data_dim*2, hid=hid,layers=layers)
        
        # Inverse model (Encoder)
        self.fz_inv = self._build_model(in_dim=data_dim, out_dim=latent_dim, hid=hid,layers=layers)
        
        self._is_fitted = False
        self._is_inverse_fitted = False

    def _build_model(self, in_dim, out_dim, hid,layers=1):
        return nn.Sequential(
            nn.Linear(in_dim, hid),
            
            *[Residual([
                nn.LayerNorm(hid),
                Prod(nn.Sequential(
                    nn.Linear(hid, hid),
                    nn.LayerNorm(hid),
                    nn.Tanh()
                )),
                nn.SiLU(),
                nn.Linear(hid, hid),
            ]) for i in range(layers)],

            nn.LayerNorm(hid),
            nn.SiLU(),
            nn.Linear(hid, out_dim),
        )

    def sample_base(self,count,device):
        half=count//2
        # efficient uniform-like space coverage sobol standard normal distribution sampler
        u = self.sobol.draw(half).to(device)           # [count, latent_dim] in [0, 1]
        z = torch.erfinv(2 * u - 1) * math.sqrt(2)      # Transform to N(0, 1)
        # reduce variance
        return torch.concat([z,-z],0)[:count]
    
    @staticmethod
    def _normal_log_prob(x, mu, log_sigma):
        #x of shape [batch,1,dim], mu of shape [1,batch,dim]
        sigma = log_sigma.exp()
        diff = (((x-mu)/sigma)**2)
        
        log_prob = -0.5 * (math.log(2 * math.pi) + 2 * log_sigma + diff)
        return log_prob

    def _get_density_params(self, z):
        """
        Passes latent z through fz to get mu and log_sigma.
        Applies the stabilization parameter C.
        """
        y = self.fz(z)
        mu, log_sigma_pred = y.chunk(2, -1)
        # Stabilization: multiply by C (starts at 0)
        log_sigma_pred = log_sigma_pred * self.C
        return mu, log_sigma_pred

    def log_prob(self, X, points_count=32,return_manifold = False):
        """
        Estimates log probability of data X using Monte Carlo integration.
        X: Tensor of shape (Batch, Data_Dim)
        """
        device = next(self.parameters()).device
        X = X.to(device)
        
        # Sample latent points
        z = self.sample_base(points_count,device)[None,:]
        
        mu, log_sigma_pred = self._get_density_params(z)
        
        # Calculate log prob for each sample point against each data point
        # X: (B, D) -> (B, 1, D)
        # mu: (1, N, D)
        logp = self._normal_log_prob(X[:, None, :], mu, log_sigma_pred).sum(-1)
        
        # LogSumExp over the latent samples to approximate marginal likelihood
        logp = torch.logsumexp(logp, dim=1) - math.log(points_count)
        if return_manifold:
            return logp,mu,z
        return logp

    def sample(self, count,sigma=1.0):
        """
        Samples from the model.
        """
        device = next(self.parameters()).device
        z = self.sample_base(count,device)
        return self.to_target(z,sigma)

    def fit(self, X, epochs=2000, batch_size=256, lr=0.01, points_count=128, print_each_n=100):
        """
        Trains the forward model (density estimator).
        X: torch tensor of shape (N, Data_Dim)
        """
        device = next(self.parameters()).device
        self.to(device)
        X = X.to(device)
        
        opt = torch.optim.AdamW(list(self.fz.parameters()) + [self.C], lr=lr,fused=True)
        sh = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        
        self.train()
        if print_each_n>0:
            print("fit marginal flow")
        running_loss = 0
        best_loss = 1e10
        best_params = None
        for i in range(1,epochs+1):
            perm = torch.randperm(X.shape[0])[:batch_size]
            batch = X[perm]
            
            opt.zero_grad(True)
            logp = self.log_prob(batch, points_count=points_count)
            loss = (-logp).mean()
            
            loss.backward()
            opt.step()
            sh.step()
            running_loss+=loss
            if i % print_each_n == 0:
                running_loss/=print_each_n
                if running_loss<best_loss:
                    best_loss=running_loss
                    best_params=deepcopy(self.state_dict())
                print(f"Epoch {i}: Loss {running_loss:.4f}")
                running_loss=0
        self.load_state_dict(best_params)
        self._is_fitted = True
        return self

    def fit_inverse(self, n_samples=10000, epochs=1000, batch_size=256, lr=0.001, weight_decay=1e-3, print_each_n=100):
        """
        Trains the inverse model (encoder).
        Maps generated data means back to the prior latent space.
        """
        if not self._is_fitted:
            raise RuntimeError("Forward model must be fitted before training inverse model.")
        
        device = next(self.parameters()).device
        self.train()
        
        # Generate synthetic training pairs for the inverse model
        # prior = torch.randn((n_samples, self.latent_dim), device=device)
        prior = self.sample_base(n_samples,device)
        
        with torch.no_grad():
            # fz maps latent -> data_params. We take the mean (mu) part.
            y = self.fz(prior)
            mu = y[:, :self.data_dim]
        
        opt = torch.optim.AdamW(self.fz_inv.parameters(), lr=lr, weight_decay=weight_decay,fused=True)
        sh = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        
        if print_each_n>0:
            print("train inverse model")
        
        running_loss = 0
        best_loss = 1e10
        best_params = None
        for i in range(1,epochs+1):
            opt.zero_grad(True)
            
            ind = torch.randperm(mu.shape[0])[:batch_size]
            batch_mu = mu[ind]
            batch_prior = prior[ind]
            
            pred = self.fz_inv(batch_mu)
            loss = nn.functional.mse_loss(pred, batch_prior)
            
            loss.backward()
            opt.step()
            sh.step()
            running_loss+=loss
            if i % print_each_n == 0:
                running_loss/=print_each_n
                
                if running_loss<best_loss:
                    best_loss = running_loss
                    best_params = deepcopy(self.state_dict())
                    
                print(f"Epoch {i}: Inv Loss {running_loss:.4f}")
                running_loss=0
        self.load_state_dict(best_params)
        self._is_inverse_fitted = True
        return self

    def to_latent(self, X):
        """
        Encodes data X into latent space using the inverse model.
        Requires fit_inverse to be called.
        """
        if not self._is_inverse_fitted:
            raise RuntimeError("Inverse model not trained. Call fit_inverse first.")
        
        self.eval()
        X = X.to(next(self.parameters()).device)
        
        z = self.fz_inv(X)
        return z

    def to_target(self, z,sigma=1.0):
        """
        Decodes latent z into data space using the forward model sampling.
        
        This is lossy operation.
        """
        self.eval()
        z = z.to(next(self.parameters()).device)
        
        # Use sampling logic
        mu, log_sigma_pred = self._get_density_params(z)
        pred = sigma*torch.randn_like(mu) * log_sigma_pred.exp() + mu
            
        return pred

    def forward(self, x, points_count=32):
        # Alias for log_prob for compatibility
        return self.log_prob(x, points_count=points_count)