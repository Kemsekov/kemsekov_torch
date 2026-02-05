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

    def log_prob(self, data, steps=None, eps=1e-2,return_separately = False):
        """
        Computes log probability using Jacobian determinant approximation via simplex volume ratios.

        This method is my attempt to port same-named method from 1-dimensional flow matching model, yet
        the resulting log-probabilities is not very useful, the pixel-wise 
        """
        model = self
        Y = data.to(model.device)

        # generate N-dimensional simplex
        simplex_points = generate_unit_simplex_vertices(self.in_channels).to(self.device)*eps

        # simplex that have some point at origin 0
        shifted_simplex=simplex_points[:-1,:]-simplex_points[-1]
        
        # log area of original simplex
        original_simplex_area_log = shifted_simplex.slogdet()[1]
        
        # make shapes match
        simplex_points = simplex_points.view(simplex_points.shape[0],1,simplex_points.shape[1],*([1]*(Y.ndim-2)),)

        # shift Y to sphere points of simplex
        Y_neighbors = Y[None,:] + simplex_points  # (B, n_neighbors, ...dim)
        Y_neighbors_shape = Y_neighbors.shape

        # Compute priors for all neighbors
        X_neighbors = self.to_prior(Y_neighbors.view(-1,*Y_neighbors_shape[2:]), steps).view(Y_neighbors_shape)

        # get area of transformed simplex
        transformed_simplex = X_neighbors[:-1,]-X_neighbors[[-1],:]
        
        transformed_simplex=transformed_simplex.permute(1,3,4,0,2)
        
        transformed_simplex_area_log = transformed_simplex.slogdet()[1]

        # area ratio is our jacobian determinant approximation
        logdet_approx = transformed_simplex_area_log - original_simplex_area_log + self.in_channels*math.log(self.in_channels)


        X = self.to_prior(Y,steps)
        prior_logp = Normal(0,1).log_prob(X).sum(1)

        if return_separately:
            return prior_logp,logdet_approx,X
        
        return prior_logp+logdet_approx