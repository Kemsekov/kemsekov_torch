import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalSpectralNorm(nn.Module):
    def __init__(self, module, name='weight', n_power_iterations=1, eps=1e-12, scale=1.0):
        """
        Wrap a module (e.g. nn.Conv2d) with bidirectional spectral normalization.
        This computes the spectral norm on the weight tensor viewed in two ways:
          (1) as (out_channels, in_channels * kernel_width * kernel_height)
          (2) as (in_channels, out_channels * kernel_width * kernel_height)
        and then averages the two estimates.
        Args:
            module (nn.Module): the module to wrap.
            name (str): name of the weight parameter (default: 'weight').
            n_power_iterations (int): number of power iterations (default: 1).
            eps (float): small epsilon for numerical stability.
            scale (float): extra scaling factor.
        """
        super(BidirectionalSpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.scale = scale

        if not self._made_params():
            self._make_params()

    def _made_params(self):
        try:
            getattr(self.module, self.name + '_orig')
            getattr(self.module, self.name + '_u')
            getattr(self.module, self.name + '_v')
            getattr(self.module, self.name + '_u2')
            getattr(self.module, self.name + '_v2')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        weight = getattr(self.module, self.name)
        # Register the original weight as a parameter under a new name
        self.module.register_parameter(self.name + '_orig', nn.Parameter(weight.data))
        # Remove the original weight parameter so that we can override it as a plain tensor
        del self.module._parameters[self.name]

        # For first reshaping: (out_channels, in_channels * kernel_width * kernel_height)
        out_channels = weight.size(0)
        weight_mat1 = weight.view(out_channels, -1)
        # Initialize u and v for weight_mat1
        u = F.normalize(weight.new_empty(out_channels).normal_(0, 1), dim=0, eps=self.eps)
        v = F.normalize(weight.new_empty(weight_mat1.size(1)).normal_(0, 1), dim=0, eps=self.eps)
        self.module.register_buffer(self.name + '_u', u)
        self.module.register_buffer(self.name + '_v', v)

        # For second reshaping: (in_channels, out_channels * kernel_width * kernel_height)
        in_channels = weight.size(1)
        weight_mat2 = weight.view(in_channels, -1)
        u2 = F.normalize(weight.new_empty(in_channels).normal_(0, 1), dim=0, eps=self.eps)
        v2 = F.normalize(weight.new_empty(weight_mat2.size(1)).normal_(0, 1), dim=0, eps=self.eps)
        self.module.register_buffer(self.name + '_u2', u2)
        self.module.register_buffer(self.name + '_v2', v2)

    def _update_u_v(self):
        # Retrieve the original weight
        weight = getattr(self.module, self.name + '_orig')
        out_channels, in_channels, k_w, k_h = weight.shape

        # First reshaping: (out_channels, in_channels * k_w * k_h)
        weight_mat1 = weight.view(out_channels, -1)
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(weight_mat1.t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.mv(weight_mat1, v), dim=0, eps=self.eps)
        sigma1 = torch.dot(u, torch.mv(weight_mat1, v))

        # Second reshaping: (in_channels, out_channels * k_w * k_h)
        weight_mat2 = weight.view(in_channels, -1)
        u2 = getattr(self.module, self.name + '_u2')
        v2 = getattr(self.module, self.name + '_v2')
        for _ in range(self.n_power_iterations):
            v2 = F.normalize(torch.mv(weight_mat2.t(), u2), dim=0, eps=self.eps)
            u2 = F.normalize(torch.mv(weight_mat2, v2), dim=0, eps=self.eps)
        sigma2 = torch.dot(u2, torch.mv(weight_mat2, v2))

        # Average the two spectral norms
        sigma = (sigma1 + sigma2) / 2.0
        # Compute normalized weight with scaling factor
        normalized_weight = self.scale * weight / (sigma + self.eps)
        # Update the module's weight attribute (as a plain tensor)
        setattr(self.module, self.name, normalized_weight)

    def forward(self, *args, **kwargs):
        self._update_u_v()
        return self.module.forward(*args, **kwargs)
