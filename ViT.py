import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from kemsekov_torch.flow_matching import FlowMatching, generate_unit_simplex_vertices
from kemsekov_torch.residual import Residual
from kemsekov_torch.attention import EfficientSpatialChannelAttention
import torch.nn as nn
from kemsekov_torch.attention import SelfAttention, EfficientSpatialChannelAttention
from kemsekov_torch.residual import Residual
from torch.distributions import Normal

class ViT(nn.Module):
    def __init__(
            self, 
            in_channels,
            hidden_dim=64,
            expand_dim=256,
            layers = 3,
            heads=8,
            head_dim=64,
            compression = 4
        ) -> None:
        super().__init__()
        
        conv = nn.Conv2d
        self.compression=compression
        self.down = nn.Sequential(
            nn.PixelUnshuffle(compression),
            conv(in_channels*(compression**2),hidden_dim,1),
        )
        
        groups = max(1,hidden_dim//32)
        if groups==1: groups=2
        
        # my self and cross attention works on conv2d-shaped tensors
        # [B,C,H,W] where H,W is actual tokens
        
        self.residuals = nn.ModuleList(
            [
                nn.ModuleList([
                    Residual([
                        SelfAttention(hidden_dim,heads=heads,head_dim=head_dim,add_rotary_embedding=True,dropout=0.0),
                        nn.GroupNorm(groups,hidden_dim),
                        nn.SiLU(),
                        Residual([
                            conv(hidden_dim,expand_dim,1),
                            nn.GroupNorm(32,expand_dim),
                            EfficientSpatialChannelAttention(expand_dim,kernel_size=3),
                            nn.SiLU(),
                            conv(expand_dim,hidden_dim,1),
                        ])
                    ],init_at_zero=False),
                    nn.Sequential(
                        nn.Linear(1,hidden_dim),
                        # nn.RMSNorm(hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim,hidden_dim*2),
                    ),
                ])
                for i in range(layers)
            ]
        )
        
        self.up = nn.Sequential(
            nn.GroupNorm(groups,hidden_dim),
            conv(hidden_dim,in_channels*(compression**2),1),
            nn.PixelShuffle(compression),
        )
        self.pos_gamma = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self,x,t):
        # x_orig = x
        while t.ndim==1:
            t = t.unsqueeze(-1)
        
        x = self.down(x)
        for r,time_emb in self.residuals:
            time_scale,time_shift = time_emb(t)[:,:,None,None].chunk(2,1)
            xt = x*(1+time_scale)+time_shift
            x = r(xt)
        return self.up(x)


def compute_subspace_log_volume(x: torch.Tensor, eps: float = 1e-8):
    """
    Computes the log-volume of the k-dimensional parallelepiped formed by 
    k vectors in N-dimensional space.
    
    Args:
        x: Tensor of shape [B, k, N]
        eps: Small constant for numerical stability in log
        
    Returns:
        log_vol: Tensor of shape [B] representing the log-volume
    """
    # 1. Transpose to [B, N, k] because QR decomposes columns
    # We want to find the volume spanned by the 'k' vectors.
    x_t = x.transpose(-1, -2)
    
    # 2. Perform QR decomposition
    # Q: [B, N, k] (orthogonal basis)
    # R: [B, k, k] (upper triangular matrix)
    # 'reduced' mode is faster and sufficient here
    Q, R = torch.linalg.qr(x_t, mode='reduced')
    
    # 3. The volume is the product of the absolute diagonal elements of R
    # We take the diagonal of the last two dimensions [B, k]
    diag_r = torch.diagonal(R, dim1=-2, dim2=-1)
    
    # 4. Compute log-volume for numerical stability
    # log(product(diag)) = sum(log(abs(diag)))
    log_vol = torch.sum(torch.log(torch.abs(diag_r) + eps), dim=-1)
    
    return log_vol

class FlowModel2d(nn.Module):
    def __init__(self, in_channels,hidden_dim = 256,attention_layers=10,head_dim=64,compression_ratio=4) -> None:
        super().__init__()
        self.register_buffer('default_steps',torch.tensor([16]))
        self.fm = FlowMatching()
        # self.fm.time_scaler = lambda x: torch.log(9*x+1)/math.log(10)
        self.in_channels=in_channels
        self.vit = ViT(
            in_channels,
            hidden_dim=hidden_dim,
            expand_dim=hidden_dim*4,
            layers=attention_layers,
            head_dim=head_dim,
            compression=compression_ratio
        )
        # self.ln = LossNormalizer2d(in_channels,64)
        self.device='cpu'
    
    def forward(self,x,t):
        return self.vit(x,t)
    
    def cuda(self,device='cuda'):
        return self.to(device)
    
    def cpu(self,device='cpu'):
        return self.to(device)
    
    def to(self,device):
        self.device = device
        return super().to(device)
    
    def to_prior(self,data : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        input_device = data.device
        return self.fm.integrate(self,data.to(self.device),steps,inverse=True).to(input_device)
    
    def to_target(self,normal_noise : torch.Tensor,steps=None):
        if not steps: steps = self.default_steps
        input_device = normal_noise.device
        return self.fm.integrate(self,normal_noise.to(self.device),steps).to(input_device)

    def sample(self,num_samples,image_size=(64,64),steps=None):
        if not steps: steps = self.default_steps
        return self.to_target(torch.randn((num_samples,self.in_channels,*image_size),device=self.device),steps)
    
    def interpolate(self,sample1,sample2,interpolation_steps):
        device = self.device
        prior1 = self.to_prior(sample1.to(device))
        prior2 = self.to_prior(sample2.to(device))

        interpolate = torch.linspace(1,0,interpolation_steps,device=device)[:,None,None,None,None]

        priors = prior1*interpolate+(1-interpolate)*prior2
        priors_shape = priors.shape
        priors=priors.view(-1,*priors_shape[2:])

        interpolation = self.to_target(priors).view(priors_shape)
        return interpolation
    
    def conditional_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        sample_shape: Tuple[int],
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 1.0,
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
        Returns:
            torch.Tensor: Samples of shape `[num_samples, input_dim]` satisfying the conditions
        
        """
        
        model = self
        model.eval()

        # Initialize z from standard normal distribution
        z = torch.randn(sample_shape, device=model.device, requires_grad=True)

        original_prior = (z * z).mean().detach()

        # Create optimizer for the latent variable z
        optimizer = torch.optim.LBFGS([z], lr=lr)
        # optimizer = torch.optim.AdamW([z], lr=lr)

        class Iteration:
            best_sample = z.clone().detach()
            best_loss = 1e8
        iteration = Iteration()
        
        def closure():
            it = iteration
            
            optimizer.zero_grad()
            # Forward pass: x = M_inv(z)
            x = model.to_target(z)

            # Compute prior loss: L_prior = ||z||² (keep z in N(0,I)) must match original generated prior
            L_prior = (z * z).mean()
            L_prior = (L_prior-original_prior)**2+mode_closeness_weight*L_prior

            # Compute constraint loss: L_constraint = constraint(x)
            L_constraint = constraint(x)

            # Total loss: L_total = L_prior + λ * L_constraint
            L_total = L_prior + L_constraint

            if L_total<it.best_loss:
                it.best_loss = L_total
                it.best_sample = z.clone().detach()
            
            L_total.backward()
            with torch.no_grad():
                z.data += (noise_scale) * torch.randn_like(z,device=model.device)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.to_target(iteration.best_sample)

        return final_x

    def log_prob(self, data, eps=1e-3,vectors_count=32):
        """
        Computes log-probability of passed data. This is very rough adaptation of exact log-prob estimation from flow_matching.py file.
        
        Accepts inputs in shape `[BATCH,C,H,W]`, returns `[BATCH]`. The larger returned value, the more likely given image to be from data distribution
        """
        model = self
        device = model.device
        Y = data.to(model.device)
        
        Y_flat = Y.flatten(1)
        
        # generate vectors on unit sphere
        vectors = torch.randn((1,vectors_count,Y_flat.shape[-1]),device=device)
        vectors = vectors*eps
        volume_before_transformation = compute_subspace_log_volume(vectors)
        
        Y_flat=Y_flat[:,None]+vectors
        Y_flat_batched = Y_flat.view(-1,Y_flat.shape[-1]) #merge added vectors and batch dimension
        Y_batched = Y_flat_batched.view(Y_flat_batched.shape[0],*Y.shape[1:]) #expand dimensions
        X_batched = self.to_prior(Y_batched).view(Y_flat_batched.shape).view(Y_flat.shape)
        
        #X_batched of shape [BATCH,vectors_count,C*H*W]
        volume_after_transformation = compute_subspace_log_volume(X_batched)
        logdet_approx = (volume_after_transformation-volume_before_transformation)/vectors_count
        # print(volume_before_transformation,volume_after_transformation)

        X = self.to_prior(Y)
        prior_logp = Normal(0,1).log_prob(X).mean([-1,-2,-3])
        return logdet_approx+prior_logp