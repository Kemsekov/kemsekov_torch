
import torch
import torch.nn as nn
class DiffusionUtils:
    def __init__(self, diffusion_timesteps: int):
        self.timesteps = torch.arange(0, diffusion_timesteps).long()
        self.time = torch.linspace(0, 1, diffusion_timesteps)

        # use cosine scheduler
        s = torch.tensor(0.0008)
        top = torch.cos((self.timesteps / diffusion_timesteps + s) * torch.pi / (2 * (1 + s)))
        bottom = torch.cos(s / (1 + s) * torch.pi / 2)
        self.alpha_bar = (top / bottom) ** 2  # cumulative product of alphas

        # Corresponding alphas
        self.alpha = torch.ones_like(self.alpha_bar)
        self.alpha[1:] = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.alpha[0] = self.alpha_bar[0]

        self.alpha_sqrt = self.alpha_bar ** 0.5
        self.one_minus_alpha_sqrt = (1 - self.alpha_bar) ** 0.5

    def diffusion_forward(self, x, t):
        ind = [None] * len(x.shape)
        ind[0] = t
        epsilon = torch.randn_like(x, device=x.device)

        alpha_bar = self.alpha_bar.to(x.device)[*ind]
        alpha_bar_sqrt = alpha_bar ** 0.5
        one_minus_alpha_bar_sqrt = (1 - alpha_bar) ** 0.5

        x_t = alpha_bar_sqrt * x + one_minus_alpha_bar_sqrt * epsilon
        return x_t, epsilon

    def diffusion_backward(self, x_t, pred_noise, t,generate_noise = False,rescale_generated_noise = False):
        """
        DDIM reverse step (η = 0): deterministic step using model-predicted noise.
        Assumes t > 0 and t is a LongTensor of shape [B].
        """
        ind = [None] * len(x_t.shape)
        ind[0] = t
        
        ind_prev = [None] * len(x_t.shape)
        ind_prev[0] = t-1

        alpha_bar = self.alpha_bar.to(x_t.device)
        alpha_bar_t = alpha_bar[*ind]
        alpha_bar_tm1 = alpha_bar[*ind_prev]  # assumes t > 0

        sqrt_alpha_bar_t = alpha_bar_t ** 0.5
        sqrt_alpha_bar_tm1 = alpha_bar_tm1 ** 0.5
        sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t) ** 0.5
        sqrt_one_minus_alpha_bar_tm1 = (1 - alpha_bar_tm1) ** 0.5

        # Estimate x0 from predicted noise
        x0_est = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
        if generate_noise:
            new_noise = torch.randn_like(pred_noise)
            if rescale_generated_noise:
                new_noise*=pred_noise.std()
                new_noise+=pred_noise.mean()
            pred_noise = new_noise
        
        # DDIM deterministic step (η = 0)
        x_prev = sqrt_alpha_bar_tm1 * x0_est + sqrt_one_minus_alpha_bar_tm1 * pred_noise
        return x_prev



def sample(diffusion_model,sample_shape,train_timesteps,inference_timesteps=20,normalize_pred = False,regenerate_noise = False,rescale_generated_noise=False):
    """
    Samples diffusion model
    Parameters:
        sample_shape: 
            Shape of sample to be denoised, like (1,in_channels,128,128)
        train_timesteps: 
            ***You must exactly specify this parameter to be same as used in training, else sampling will be broken!***
        inference_timesteps: 
            Timesteps for sampling
        normalize_pred: normalize predicted noise to be mean 0 std 1
        regenerate_noise: to regenerate noise each denoising step or not.
        rescale_generated_noise: rescale generated noise to have same std and mean as predicted noise
    Last three parameters for some datasets helps to generate better samples, but there is no single best configuration for them, play around with these three parameters
    """
    diff_util = DiffusionUtils(inference_timesteps)
    next_t = torch.randn(sample_shape,device=list(diffusion_model.parameters())[0].device)

    # plt.figure(figsize=(12,12))
    for t in reversed(diff_util.timesteps[1:]):
        T = torch.tensor([t]*next_t.shape[0])
        with torch.no_grad():
            T = torch.round(T*train_timesteps/inference_timesteps).long()
            pred_noise_ = diffusion_model(next_t,T)
            if normalize_pred:
                pred_noise_/=pred_noise_.std()
                pred_noise_-=pred_noise_.mean()

        t=t.item()
        next_t = diff_util.diffusion_backward(next_t,pred_noise_,t,generate_noise=regenerate_noise,rescale_generated_noise=rescale_generated_noise)

    return next_t

from kemsekov_torch.residual import Residual, ResidualBlock
from kemsekov_torch.attention import *
from kemsekov_torch.rotary_emb import RotaryEmbInplace
from kemsekov_torch.positional_emb import PositionalEncodingPermute,ConcatPositionalEmbeddingPermute

class TimeContextEmbedding(torch.nn.Module):
    def __init__(self,in_channels,max_timesteps=64):
        """
        Layer that combines input x, timestep and context.
        
        in_channels: input channels
        
        context_channels: channels of context passed
        
        attn_heads: cross-attention heads count
        
        dimensions: input/context spatial dimensions count
        
        max_timesteps: this parameter defines max timestep input value.
        
        """
        super().__init__()
        # self.context2down1 = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1](context_channels,in_channels,kernel_size=1)
        # self.combine_x_time = TransformerDecoderLayerMultidim(in_channels,dim_feedforward=in_channels,nhead=attn_heads)
        
        x=torch.randn((1,in_channels,max_timesteps,1))
        time = PositionalEncodingPermute(in_channels,freq=max_timesteps)(x)[0].transpose(0,1)[...,0]
        self.time_emb = torch.nn.Parameter(time)
        
    def forward(self,x,timestep):
        """
        x: input of shape (B,C,...) where ... is spatial dimensions
        timestep: long tensor of shape (B). It is a timestep for each element in batch.
        """
        time = self.time_emb[timestep.cpu()].to(x.device)
        dims = list(time.shape)+[1]*(len(x.shape)-2)
        time=time.view(dims)
        x_t = x+time.expand_as(x)
        return x_t

class DiffusionBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,transformer_blocks = 2,attn_heads=8,max_timesteps=64,normalization='batch',dimensions=2):
        super().__init__()
        mlp_dim = 2*out_channels
        self.down = ResidualBlock(
            in_channels,
            [mlp_dim,out_channels],
            kernel_size=4,
            stride=2,
            dimensions=dimensions,
            activation=nn.GELU,
            normalization=normalization
        )
        
        self.embed_context_to_down = TimeContextEmbedding(out_channels,max_timesteps)
        self.sa = Residual([
            RotaryEmbInplace(out_channels),
            ConcatPositionalEmbeddingPermute(out_channels,256,dimensions=dimensions),
            *[
                FlattenSpatialDimensions([
                    # TransformerSelfAttentionBlock(out_channels,attn_heads,mlp_dim),
                    LinearSelfAttentionBlock(out_channels,mlp_dim,attn_heads)
                ])
                for i in range(transformer_blocks)
            ]
        ])
        
    def forward(self,x,time):
        xt = self.down(x)
        xt = self.embed_context_to_down(xt,time)
        return self.sa(xt)
    
    def transpose(self):
        self.down = self.down.transpose()
        return self

class Diffusion(torch.nn.Module):
    def __init__(self, in_channels,max_timesteps=512):
        super().__init__()
        
        common_diff_block = dict(
            normalization='group',
            attn_heads = 16,
            dimensions = 2,
            max_timesteps=max_timesteps
        )
        
        commin_res_block=dict(
            normalization=common_diff_block['normalization'],
            dimensions=common_diff_block['dimensions'],
            activation = nn.GELU
        )
        
        self.dimensions = common_diff_block['dimensions']
        self.upscale_input = ResidualBlock(
            in_channels,
            64,
            kernel_size=3,
            **commin_res_block
        )
        
        self.down1 = DiffusionBlock(64,128,**common_diff_block)
        self.down2 = DiffusionBlock(128,128,**common_diff_block)
        self.down3 = DiffusionBlock(128,256,**common_diff_block)
        self.down4 = DiffusionBlock(256,512,**common_diff_block)
        
        self.up1 = DiffusionBlock(512,256,**common_diff_block).transpose()
        self.merge_up1_down3 = ResidualBlock(512,[256,256],kernel_size=1,**commin_res_block)
        
        self.up2 = DiffusionBlock(256,128,**common_diff_block).transpose()
        self.merge_up2_down2 = ResidualBlock(256,[128,128],kernel_size=1,**commin_res_block)
        
        self.up3 = DiffusionBlock(128,128,**common_diff_block).transpose()
        self.merge_up3_down1 = ResidualBlock(256,[128,128],kernel_size=1,**commin_res_block)

        self.up4 = DiffusionBlock(128,64,transformer_blocks=0,**common_diff_block).transpose()
        self.merge_up4_x = ResidualBlock(128,[64,64],kernel_size=1,**commin_res_block)
        
        # to produce proper logits, combine model output that input
        self.final = [torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d][self.dimensions-1](64,in_channels,kernel_size=1)
    
    # x is batched single example with all noise levels
    # timestep is indices of noise levels for each sample is x
    # context_embedding is batched tensor with
    def forward(self,x, timestep : torch.LongTensor):
        """
        x: [batch,channels,width,height]
        
        timestep: [batch] tensor of type long
        
        context: [batch,context_channels, d1, d2]. d1 d2 can be of any size
        """
        x = self.upscale_input(x)
        d1 = self.down1(x,timestep)
        d2 = self.down2(d1,timestep)
        d3 = self.down3(d2,timestep)
        d4 = self.down4(d3,timestep)
        
        u1 = self.up1(d4,timestep)
        u1 = self.merge_up1_down3(torch.concat([u1,d3],1))
        
        u2 = self.up2(u1,timestep)
        u2 = self.merge_up2_down2(torch.concat([u2,d2],1))
        
        u3 = self.up3(u2,timestep)
        u3 = self.merge_up3_down1(torch.concat([u3,d1],1))
        
        u4 = self.up4(u3,timestep)
        x = self.merge_up4_x(torch.concat([u4,x],1))
        
        return self.final(x)
    