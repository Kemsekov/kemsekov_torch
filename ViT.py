import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.common_modules import Prod, AddConst
from kemsekov_torch.flow_matching import FlowMatching, generate_unit_simplex_vertices
from kemsekov_torch.residual import Residual
from kemsekov_torch.attention import EfficientSpatialChannelAttention
import torch.nn as nn
from kemsekov_torch.attention import SelfAttention, EfficientSpatialChannelAttention, zero_module
from kemsekov_torch.residual import Residual
from torch.distributions import Normal

class ViT(nn.Module):
    def __init__(
            self, 
            in_channels,
            hidden_dim=512,
            layers = 3,
            head_dim=64,
            compression = 4,
            dropout=0.0
        ) -> None:
        super().__init__()
        groups = max(1,hidden_dim//32)
        if groups==1: groups=2
        
        conv = nn.Conv2d
        self.compression=compression
        self.down = nn.Sequential(
            nn.PixelUnshuffle(compression),
            conv(in_channels*(compression**2),hidden_dim,1),
        )

        # my self and cross attention works on conv2d-shaped tensors
        # [B,C,H,W] where H,W is actual tokens
        act = nn.SiLU
        self.residuals = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    # self-attention already returns init-zeroed residual
                    Residual([
                        SelfAttention(
                            hidden_dim,
                            heads=min(4,hidden_dim//head_dim), # use at least 4 heads
                            head_dim=head_dim,
                            add_rotary_embedding=True,
                            dropout=dropout,
                            output_bias=False,
                            add_absolute_pos=True
                        ),
                        nn.GroupNorm(32,hidden_dim),
                        EfficientSpatialChannelAttention(hidden_dim,kernel_size=3),
                        act(),
                        zero_module(conv(hidden_dim,hidden_dim,1,bias=False)),
                    ],init_at_zero=False),
                ),
                nn.Sequential(
                    nn.Linear(1,hidden_dim),
                    Prod(nn.Sequential(
                        nn.RMSNorm(hidden_dim),
                        nn.Linear(hidden_dim,hidden_dim),
                        nn.Tanh(),
                    )),
                    act(),
                    zero_module(nn.Linear(hidden_dim,hidden_dim*2)),
                )
            ])
            for i in range(layers)
        ])
        
        self.up = nn.Sequential(
            nn.GroupNorm(groups,hidden_dim),
            EfficientSpatialChannelAttention(hidden_dim,kernel_size=3),
            conv(hidden_dim,in_channels*(compression**2),1),
            nn.PixelShuffle(compression),
        )
        # self.orig_x_gamma = nn.Sequential(
        #     nn.Linear(1,32),
        #     act(),
        #     zero_module(nn.Linear(32,in_channels)),
        # )
        
        # self.orig_x_gamma = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self,x,t):
        x_orig = x
        while t.ndim==1:
            t = t.unsqueeze(-1)
        
        x = self.down(x)
        for r,time_emb in self.residuals:
            time_scale,time_shift = time_emb(t)[:,:,None,None].chunk(2,1)
            xt = x*(1+time_scale)+time_shift
            x = r(xt)
        return self.up(x)

class LossNormalizer2d(nn.Module):
    def __init__(self, in_channels,hidden_dim) -> None:
        super().__init__()
        
        hid1 = hidden_dim//4
        hid2 = hidden_dim//2
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels,hid1,bias=False,kernel_size=4,stride=2),
            nn.GroupNorm(max(4,hid1//32),hid1),
            nn.SiLU()
        )
        
        self.down2 = Residual([
            nn.Conv2d(hid1,hid2,bias=False,kernel_size=4,stride=2),
            nn.GroupNorm(max(4,hid2//32),hid2),
            nn.SiLU()
        ])
        
        self.down3 = Residual([
            nn.Conv2d(hid2,hidden_dim,bias=False,kernel_size=4,stride=2),
            nn.GroupNorm(max(4,hidden_dim//32),hidden_dim),
            nn.SiLU()
        ])
        
        self.down4 = Residual([
            nn.Conv2d(hidden_dim,hidden_dim*2,bias=False,kernel_size=4,stride=2),
            nn.GroupNorm(max(4,hidden_dim*2//32),hidden_dim*2),
            nn.SiLU()
        ])
        self.time_emb = nn.Sequential(
            nn.Linear(1,hidden_dim),
            Prod(nn.Sequential(
                nn.RMSNorm(hidden_dim),
                nn.Linear(hidden_dim,hidden_dim),
                nn.Tanh(),
            )),
            nn.SiLU(),
            zero_module(nn.Linear(hidden_dim,hid1)),
        )
        
        self.out = nn.Conv2d(hidden_dim*2,1,kernel_size=1)
    
    def forward(self,x,time):
        B = x.shape[0]
        if time.ndim==1: time=time[:,None]
        time = self.time_emb(time).view(B,-1,1,1)
        
        x = self.down1(x)
        x=x+time
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        return self.out(x).mean([-1,-2],keepdim=True)
        

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
            layers=attention_layers,
            head_dim=head_dim,
            compression=compression_ratio
        )
        self.ln = LossNormalizer2d(in_channels,256)
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
    
    def reflow_loss_and_metric(self,x0 : torch.Tensor,images : torch.Tensor):
        """
        Computes loss and metrics for reflowing FM model from known latents (x0) and images
        """
        model = self
        images_pred = model.to_target(x0)
        x0_pred = model.to_prior(images)
        
        loss = F.mse_loss(images,images_pred)
        loss_inv = F.mse_loss(x0,x0_pred)
        
        r2_t = r2_score(images_pred,images)
        r2_p = r2_score(x0_pred,x0)
        return loss+loss_inv,{
            "r2_target":r2_t,
            "r2_prior":r2_p,
            'r2':(r2_t+r2_p)/2
        }
        
    def train_loss_and_metric(self,images : torch.Tensor,contrastive_loss_weight=0.1,loss_normalization_power=0):
        """
        Computes loss and metric for given images batch.
        images: of shape [B,C,H,W]
        loss_normalization_power: when set to 0, no loss normalization is applied, when set to 1, model tries to exactly predict per-sample loss
        """
        
        model = self
        ln = model.ln
        
        x0 = torch.randn_like(images)
        pred_dir,target_dir,contrast_dir,t = model.fm.contrastive_flow_matching_pair(
            model,
            x0,
            images
        )
        pred_loss = F.mse_loss(pred_dir,target_dir,reduction='none')
        contrastive_loss = F.mse_loss(pred_dir,contrast_dir,reduction='none')
        # make it negative
        contrastive_loss-=contrastive_loss.max().detach()+1e-4
        contrastive_loss=contrastive_loss/contrastive_loss.abs().mean().detach()*pred_loss.abs().mean().detach()
        
        # scale it
        contrastive_loss = contrastive_loss_weight*contrastive_loss
        
        # sample-wise loss
        sample_loss = pred_loss-contrastive_loss
        
        if loss_normalization_power>0:
            pred_log_loss = ln(target_dir,t)
            loss_normalizer_target = -(sample_loss.detach()+1e-4).log()
            loss_normalizer_loss = F.mse_loss(pred_log_loss,loss_normalizer_target)
            weight = pred_log_loss.exp().detach()**loss_normalization_power
        else:
            weight = 1
            loss_normalizer_loss=0  
        #scale loss by it's prediction
        weighed_loss = (weight*sample_loss).mean()
        loss = weighed_loss+loss_normalizer_loss
        
        return loss,{
            "r2":r2_score(pred_dir,target_dir),
            "r2_loss":r2_score(pred_log_loss,loss_normalizer_target) if loss_normalization_power>0 else 0
        }