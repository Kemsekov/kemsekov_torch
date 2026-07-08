"""
File for cycle-gan implementations
"""
from dataclasses import dataclass
from typing import Callable
from kemsekov_torch.common_modules import GradientReversal
import torch
import torch.nn as nn
from kemsekov_torch.ema_pytorch.ema_pytorch import EMA
from copy import deepcopy
from torch.utils.data import Dataset

def generator_score(x):
    return (x).mean()

def critic_loss(
    critic : torch.nn.Module,
    real_data : torch.Tensor,
    fake_data : torch.Tensor):
    
    fake_scores = generator_score(critic(fake_data))
    
    if torch.is_grad_enabled():    
        alpha_shape = [real_data.shape[0]] + [1] * (real_data.ndim - 1)
        alpha = torch.rand(alpha_shape, device=real_data.device,dtype=real_data.dtype)

        interpolated_samples = (
            alpha * real_data +
            (1 - alpha) * fake_data
        )
        interpolated_scores = critic(interpolated_samples)
        # Compute gradients w.r.t. interpolated samples
        gradients = torch.autograd.grad(
            outputs=interpolated_scores.sum(),  # Scalar output for grad
            inputs=interpolated_samples,
            create_graph=True,  # Essential for higher-order gradients
            retain_graph=True
        )[0]
        dims = list(range(gradients.ndim))[1:]
        grad_l2_norm = gradients.pow(2).sum(dim=dims).sqrt() # per sample l2 norm
        grad_penalty=(grad_l2_norm-1).pow(2).mean()
    else:
        grad_penalty=fake_scores*0
    real_scores = generator_score(critic(real_data))
    loss = fake_scores-real_scores
    
    return loss,fake_scores,real_scores, grad_penalty

def generator_loss(critic,generated_data):
    generated_score = generator_score(critic(generated_data))
    return (-generated_score)

@dataclass
class CycleGanForward:
    """
    Container for CycleGAN forward pass outputs with detailed loss components and intermediate tensors.
    """
    cycle_consistency_loss: torch.Tensor
    """
        Loss of cycle mapping of domains to each other.\n\n
        How good generators at preserving domains when applied to each other\n\n
        If `b'=generator_ab(a)` and `a'=generator_ba(b')`, then if `a` is similar to `a'`, the loss is small
    """
    identity_loss: torch.Tensor
    """
        Loss of identity mapping from generators.\n\n
        How good generators at keeping other domain:\n\n 
        `generator_ab(b)` same as `b`, `generator_ba(a)` same as `a`
    """
    adversarial_loss: torch.Tensor
    """
        Adversarial loss part of cycle gan.\n\n
        Forces critics to distinguish real and fake samples.\n\n
        It computed as mean `critic(fake)-critic(real)` of both critics.\n\n
    """
    reconstruction_a : torch.Tensor
    """
        Samples from A domain was mapped to domain B via `b=generator_ab(a)`, then it was reconstructed by another
        generator `a_rec=generator_ba(b)`.
        So this tensor should be similar to batch from domain A
    """
    reconstruction_b : torch.Tensor
    """
        Samples from B domain was mapped to domain A via `a=generator_ba(b)`, then it was reconstructed by another
        generator `b_rec=generator_ab(a)`.
        So this tensor should be similar to batch from domain B
    """
    identity_a : torch.Tensor
    """
        Results of `generator_ba(a)`. For foreign domain, generator should return value unchanged. 
        So this tensor should be similar to batch of domain A
    """
    identity_b : torch.Tensor
    """
        Results of `generator_ab(b)`. For foreign domain, generator should return value unchanged. 
        So this tensor should be similar to batch of domain B
    """
    gradient_penalty: torch.Tensor
    """
        Gradient penalty part of WGAN-CP loss function.\n\n
        It is computed as `mse(l2_norm(d/dx critic(x)),1)`\n\n
        where `x=alpha*real+(1-alpha)*fake` for random uniform `alpha`\n\n
        Adding this term to loss function forces the Lipschitz constraint of critic to be around 1.\n\n
        It forces the first derivative of critics between real and fakes samples to have l2 norm about 1, so our critic is well-behaved, not too smooth, not too steep
    """
    fake_scores_a : torch.Tensor
    """
        Output of `critic_a(fake_a)` where `fake_a=generator_ba(b)`. \n\n
        We expect it to be `inf`
    """
    fake_scores_b : torch.Tensor
    """
        Output of `critic_b(fake_b)` where `fake_b=generator_ab(a)`. \n\n
        We expect it to `inf`
    """
    real_scores_a : torch.Tensor
    """
        Output of `critic_a(a)` where `a` is samples from A domain. \n\n
        We expect it to be `-inf`
    """
    real_scores_b : torch.Tensor
    """
        Output of `critic_b(b)` where `b` is samples from B domain. \n\n
        We expect it to be `-inf`
    """
    
    def loss(
        self,
        cycle_consistency_lambda=10.0,
        identity_lambda=5.0,
        adversarial_lambda=1.0,
        gradient_penalty=5.0
    ):
        """
        Computes loss function for all generators and critics at the same time.
        
        cycle_consistency_lambda: Affects generators. How strong generators should keep cycle consistency
        identity_lambda: Affects generators. How important for us to keep identity mappings
        adversarial_lambda: Affects critics. How strong updates to critics should be.
        gradient_penalty: Forces Lipschitz constraint of critic to be around 1. Larger values will enforce larger smoothness of critics
        """
        return \
            self.cycle_consistency_loss*cycle_consistency_lambda+\
            self.identity_loss*identity_lambda+\
            self.adversarial_loss*adversarial_lambda+\
            self.gradient_penalty*gradient_penalty

class CycleGan(torch.nn.Module):
    """
    Modular Cycle-Consistent Adversarial Network with factory-based architecture injection.
    
    This implementation differs from ordinary cycle-gan, by keeping all training process within single optimizer
    and single backward step.
    
    This implementation decouples network architecture from training logic by accepting
    generator and critic factory functions. Features:
    
    1. **Flexible Architecture**:
       - Generators and critics instantiated via provided factory functions
       - Supports any architecture conforming to input/output shape requirements
    
    2. **Wasserstein Critics with GP**:
       - Uses gradient penalty (WGAN-GP) for stable adversarial training
       - Gradient reversal layer enables single-backward-pass training
    
    3. **Core Loss Components**:
       - Cycle consistency loss (L1) for structural preservation
       - Identity loss (L1) for domain feature preservation
       - Adversarial loss with gradient penalty
    
    4. **Transparent Outputs**:
       - Returns structured CycleGanForward dataclass with all intermediate tensors
       - Enables custom loss weighting and detailed monitoring
    
    Input Requirements:
    ------------------
    - Input tensors must have shape: [batch_size, channels, sequence_length]
    - For point clouds (e.g., 2D coordinates): sequence_length=1 → [B, 2, 1]
    - Generators must accept [B, C, L] and return same-shaped tensors
    - Critics must accept [B, C, L] and return [B, 1, L] or [B, 1] logits
    
    Critical Implementation Notes:
    -----------------------------
    1. **Critic Requirements**:
       - MUST output unbounded logits (no sigmoid/tanh final activation)
       - Optimal critic behavior: real → -∞, fake → +∞
       - Gradient penalty enforces 1-Lipschitz constraint
    
    2. **Gradient Reversal**:
       - During critic evaluation on fake samples, gradients are reversed
       - This allows generator updates through critic gradients in single backward pass
       - Implemented via kemsekov_torch's GradientReversal layer
    """
    def __init__(
        self,
        get_generator : Callable[[],nn.Module],
        get_critic : Callable[[],nn.Module]
    ):
        """
        Parameters
        ----------
        get_generator: 
            Method to create generator. 
            Generator must accept and returns same-shaped tensors.
            Generator is a module that accepts batch samples of one domain, and return batch of another domain
        get_critic:
            Method to create critics.\n
            Critics must accepts batch of samples of shape [B,...] and for each batch return logits of any shape [B,...].\n
            The returned values of critic will be averaged.\n
            If sample is fake, `critic(fake) -> inf`\n
            If sample is real, `critic(real) -> -inf`
        """
        super().__init__()
        
        # simple generators
        self.g_ab = get_generator()
        self.g_ba = get_generator()
        
        # we need surely that out critics are returning logits, not some bounded values, else training will not work
        # critic -> -inf for real samples, critic -> inf for fake samples
        self.critic_a = get_critic()
        self.critic_b = get_critic()
        
        self.grl = GradientReversal()

    def forward(self,batch_a,batch_b):
        # identity loss
        identity_a = self.g_ba(batch_a)
        identity_b = self.g_ab(batch_b)
        identity_loss = (
            torch.nn.functional.l1_loss(batch_a,identity_a)+
            torch.nn.functional.l1_loss(batch_b,identity_b)
        )
        
        # generate fake samples of other domain
        fake_a = self.g_ba(batch_b)
        fake_b = self.g_ab(batch_a)
        
        # cycle loss
        rec_a = self.g_ba(fake_b)
        rec_b = self.g_ab(fake_a)
        cycle_consistency_loss = (
            torch.nn.functional.l1_loss(batch_a,rec_a)+
            torch.nn.functional.l1_loss(batch_b,rec_b)
        )
        
        # # if sample is fake, critic -> 1, if true critic -> 0 
        c1,fake_scores_a,real_scores_a,grad_penalty_a = critic_loss(
            self.critic_a,
            batch_a,
            self.grl(fake_a),
        )
        c2,fake_scores_b,real_scores_b,grad_penalty_b = critic_loss(
            self.critic_b,
            batch_b,
            self.grl(fake_b),
        )
        
        adversarial_loss = (c1+c2)
        grad_penalty = (grad_penalty_a+grad_penalty_b)
        return CycleGanForward(
            cycle_consistency_loss=cycle_consistency_loss,
            identity_loss=identity_loss,
            adversarial_loss=adversarial_loss,
            reconstruction_a=rec_a,
            reconstruction_b=rec_b,
            identity_a=identity_a,
            identity_b=identity_b,
            gradient_penalty=grad_penalty,
            fake_scores_a=fake_scores_a,
            fake_scores_b=fake_scores_b,
            real_scores_a=real_scores_a,
            real_scores_b=real_scores_b,
        )

class UnpairedDataset(Dataset):
    def __init__(self, domain1_data,domain2_data):
        super().__init__()
        # Collect all image files recursively from folder1 and folder2
        self.domain1_data = domain1_data
        self.domain2_data = domain2_data
        
    def __len__(self):
        """Return the minimum number of images between the two folders."""
        return max(len(self.domain1_data), len(self.domain2_data))
    
    def __getitem__(self, index):
        """Return a pair of images from folder1 and folder2."""
        i1 = self.domain1_data[index % len(self.domain1_data)]
        i2 = self.domain2_data[index % len(self.domain2_data)]
        
        return i1,i2

def get_optim_groups(model, weight_decay=1e-2):
    decay_params = set()
    no_decay_params = set()
    norm_layers=(
        nn.LayerNorm, 
        nn.RMSNorm, 
        nn.BatchNorm1d, 
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    def process_model(m):
        for mn, module in m.named_modules():
            # recurse=False ensures we only process parameters directly belonging to this module
            for pn, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                
                # Rule 1: Biases should NEVER be decayed
                if pn.endswith('bias'):
                    no_decay_params.add(p)
                # Rule 2: Normalization layer weights should NEVER be decayed
                elif isinstance(module, norm_layers):
                    no_decay_params.add(p)
                # Rule 3: Protect your custom time_scaler from being shrunk to 0
                elif 'scaler' in pn.lower():
                    no_decay_params.add(p)
                # Rule 4: Everything else (Linear weights, etc.) gets weight decay
                else:
                    decay_params.add(p)

    process_model(model)
    return [
        {"params": list(decay_params), "weight_decay": weight_decay},
        {"params": list(no_decay_params), "weight_decay": 0.0}
    ]


def train_cycle_gan(
    cycle_gan:CycleGan,
    dataloader:torch.utils.data.DataLoader,
    epochs=10,
    lr=0.1,
    best_checkpoint_loss_history_size=16,
    ema_beta=0.995,
    loss_lambda=0.5,
    verbose=False,
    max_grad_norm=1.0
):
    """
    Trains a CycleGAN model using a single optimizer and optional EMA stabilization.

    The training loop performs joint updates of generators and critics using the
    combined CycleGAN loss (cycle consistency, identity, adversarial loss, and
    gradient penalty). The best model state is tracked using a moving average of
    recent losses and restored after training.

    Parameters
    ----------
    cycle_gan:
        Initialized CycleGan model to train.

    dataloader:
        DataLoader providing unpaired batches from both domains as `(domain_a, domain_b)`.

    epochs:
        Number of complete passes through the dataset.

    lr:
        Learning rate for AdamW optimizer.

    best_checkpoint_loss_history_size:
        Number of recent loss values used for selecting the best checkpoint.
        Larger values make checkpoint selection less sensitive to noise.

    ema_beta:
        Exponential moving average coefficient for generator weights.
        Set to 0 to disable EMA.

    loss_lambda:
        Target loss value used by absolute loss stabilization:
        `abs(loss - loss_lambda)`.

    verbose:
        If True, prints device information and best checkpoint updates.

    max_grad_norm:
        Maximum gradient norm used for gradient clipping.
        Set to None to disable gradient clipping.

    Returns
    -------
    CycleGan:
        The trained model with weights restored to the best observed checkpoint.
    """
    param = list(cycle_gan.parameters())[0]
    device = param.device
    dtype = param.dtype
    if verbose:
        print("Device",device)
        print("Dtype",dtype)
    
    total_steps=len(dataloader)*epochs
    opt = torch.optim.AdamW(get_optim_groups(cycle_gan),lr=lr,fused=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_steps)
    gen_ema1 = EMA(cycle_gan.g_ab,beta=ema_beta,power=1)
    gen_ema2 = EMA(cycle_gan.g_ba,beta=ema_beta,power=1)

    running_loss = []
    best_loss = 1e10
    best_model = None

    for i in range(epochs):
        if ema_beta>0:
            gen_ema1.update_model_with_ema()
            gen_ema2.update_model_with_ema()
        for a,b in dataloader:
            opt.zero_grad(True)
            l = cycle_gan(a.to(device,dtype=dtype),b.to(device,dtype=dtype))
            loss = (l.loss()-loss_lambda).abs()
            loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    cycle_gan.parameters(),
                    max_grad_norm
                )
            
            running_loss.append(loss.item())
            running_loss=running_loss[-best_checkpoint_loss_history_size:]
            
            mean_running = sum(running_loss)/best_checkpoint_loss_history_size
            
            if len(running_loss)==best_checkpoint_loss_history_size and mean_running<best_loss:
                best_loss=mean_running
                best_model=deepcopy(cycle_gan.state_dict())
                if verbose: print(f"Epoch {i}\tLoss: {mean_running:0.4f}")
                if ema_beta>0:
                    gen_ema1.update()
                    gen_ema2.update()
            
            opt.step()
            sch.step()
            
    if best_model is not None:
        cycle_gan.load_state_dict(best_model)
    return cycle_gan