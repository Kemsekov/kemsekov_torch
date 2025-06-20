
import torch
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
