"""
File for cycle-gan implementations
"""
from dataclasses import dataclass
from typing import Callable
from kemsekov_torch.common_modules import GradientReversal
import torch
import torch.nn as nn

def get_square_mean_signed(x):
    return (x).mean()

def critic_loss(
    critic : torch.nn.Module,
    real_data : torch.Tensor,
    fake_data : torch.Tensor):
    
    fake_scores = get_square_mean_signed(critic(fake_data))
    
    if torch.is_grad_enabled():    
        noise = torch.rand(fake_data.shape,device=real_data.device)
        interpolated_samples = (real_data*noise+(1-noise)*fake_data).detach().requires_grad_(True)
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
    real_scores = get_square_mean_signed(critic(real_data))
    loss = fake_scores-real_scores
    
    return loss,fake_scores,real_scores, grad_penalty

def generator_loss(critic,generated_data):
    generated_score = get_square_mean_signed(critic(generated_data))
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
        gradient_penalty=1.0
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