import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseBSN(nn.Module):
    def __init__(self, module, n_power_iterations=1, eps=1e-12):
        super().__init__()
        self.module = module
        weight = self.module.weight  # Direct access to weight
        fan_in, fan_out = self._get_fan_in_fan_out(weight)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.scaling_factor = (2 / fan_in) ** 0.5  # Kaiming scaling
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.u_forward = None
        self.v_forward = None
        self.u_backward = None
        self.v_backward = None
        self.has_bias = hasattr(self.module, 'bias')
        self.is_conv = isinstance(self.module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        self.is_conv_t = isinstance(self.module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
        self.power_iteration(module.weight)

    def power_iteration(self, weight):
        """Compute the bidirectional spectral norm using power iteration."""
        if self.is_conv:
            # Standard spectral norm for Conv layers
            weight_mat_forward = weight.view(weight.size(0), -1)  # (c_out, c_in * k_w * k_h)
            weight_mat_backward = weight.view(-1, weight.size(1))  # (c_out * k_w * k_h, c_in)
        elif self.is_conv_t:
            # For ConvTranspose, weight is (in_channels, out_channels, kernel_size, ...)
            weight_mat_forward = weight.view(weight.size(0), -1)  # (in_channels, out_channels * k_w * k_h)
            weight_mat_backward = weight.view(-1, weight.size(1))  # (in_channels * k_w * k_h, out_channels)
        else:  # Linear layers
            weight_mat_forward = weight.view(weight.size(0), -1)  # (out_features, in_features)
            weight_mat_backward = weight.view(-1, weight.size(0))  # (in_features, out_features)

        # Initialize u and v for both forward and backward if not already initialized
        if self.u_forward is None or self.v_forward is None:
            self.u_forward = torch.randn(weight_mat_forward.size(0), device=weight_mat_forward.device, dtype=weight_mat_forward.dtype)
            self.v_forward = torch.randn(weight_mat_forward.size(1), device=weight_mat_forward.device, dtype=weight_mat_forward.dtype)
            self.u_backward = torch.randn(weight_mat_backward.size(0), device=weight_mat_backward.device, dtype=weight_mat_backward.dtype)
            self.v_backward = torch.randn(weight_mat_backward.size(1), device=weight_mat_backward.device, dtype=weight_mat_backward.dtype)

        # Power iteration for forward spectral norm
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                self.v_forward = F.normalize(weight_mat_forward.t() @ self.u_forward, dim=0, eps=self.eps)
                self.u_forward = F.normalize(weight_mat_forward @ self.v_forward, dim=0, eps=self.eps)
            sigma_forward = torch.dot(self.u_forward, weight_mat_forward @ self.v_forward)

        # Power iteration for backward spectral norm
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                self.v_backward = F.normalize(weight_mat_backward.t() @ self.u_backward, dim=0, eps=self.eps)
                self.u_backward = F.normalize(weight_mat_backward @ self.v_backward, dim=0, eps=self.eps)
            sigma_backward = torch.dot(self.u_backward, weight_mat_backward @ self.v_backward)

        # Combine into bidirectional spectral norm
        sigma = torch.sqrt(sigma_forward ** 2 + sigma_backward ** 2)
        return sigma

    def _get_fan_in_fan_out(self, weight):
        """Compute fan_in and fan_out based on weight shape."""
        if weight.dim() == 2:  # Linear layers
            fan_in, fan_out = weight.size(1), weight.size(0)
        elif weight.dim() >= 3:  # Convolutional layers (1D, 2D, 3D)
            receptive_field_size = 1
            for s in weight.shape[2:]:
                receptive_field_size *= s
            fan_in = weight.size(1) * receptive_field_size
            fan_out = weight.size(0) * receptive_field_size
        else:
            raise ValueError("Unsupported weight shape")
        return fan_in, fan_out

    def forward(self, x):
        weight = self.module.weight  # Direct access to weight
        sigma = self.power_iteration(weight)
        W_sn = weight / sigma  # Spectral normalization
        W_b = self.gamma * self.scaling_factor * W_sn  # Bidirectional scaling

        if self.has_bias:
            return self._apply_functional(x, W_b, self.module.bias)
        return self._apply_functional(x, W_b, None)

    def _apply_functional(self, x, weight, bias):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError

class LinearBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.linear(x, weight, bias)

class Conv1dBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.conv1d(
            x, weight, bias,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups
        )

class Conv2dBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.conv2d(
            x, weight, bias,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups
        )

class Conv3dBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.conv3d(
            x, weight, bias,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups
        )

class ConvTranspose1dBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.conv_transpose1d(
            x, weight, bias,
            stride=self.module.stride,
            padding=self.module.padding,
            output_padding=self.module.output_padding,
            groups=self.module.groups,
            dilation=self.module.dilation
        )

class ConvTranspose2dBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.conv_transpose2d(
            x, weight, bias,
            stride=self.module.stride,
            padding=self.module.padding,
            output_padding=self.module.output_padding,
            groups=self.module.groups,
            dilation=self.module.dilation
        )

class ConvTranspose3dBSN(BaseBSN):
    def _apply_functional(self, x, weight, bias: torch.Tensor | None):
        return F.conv_transpose3d(
            x, weight, bias,
            stride=self.module.stride,
            padding=self.module.padding,
            output_padding=self.module.output_padding,
            groups=self.module.groups,
            dilation=self.module.dilation
        )

def _BidirectionalSpectralNormalization(module, n_power_iterations=1, eps=1e-12):
    """
    Applies bidirectional spectral normalization to the given module with hardcoded 'weight'.
    """
    if isinstance(module, nn.Linear):
        return LinearBSN(module, n_power_iterations, eps)
    elif isinstance(module, nn.Conv1d):
        return Conv1dBSN(module, n_power_iterations, eps)
    elif isinstance(module, nn.Conv2d):
        return Conv2dBSN(module, n_power_iterations, eps)
    elif isinstance(module, nn.Conv3d):
        return Conv3dBSN(module, n_power_iterations, eps)
    elif isinstance(module, nn.ConvTranspose1d):
        return ConvTranspose1dBSN(module, n_power_iterations, eps)
    elif isinstance(module, nn.ConvTranspose2d):
        return ConvTranspose2dBSN(module, n_power_iterations, eps)
    elif isinstance(module, nn.ConvTranspose3d):
        return ConvTranspose3dBSN(module, n_power_iterations, eps)
    return module

from kemsekov_torch.common_modules import wrap_submodules

def BidirectionalSpectralNormalization(
    module,
    layers_to_apply=[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear],
    n_power_iterations=1,
    eps=1e-12
):
    """
    Recursively applies bidirectional spectral normalization to all submodules listed in `layers_to_apply`.
    """
    for l in layers_to_apply:
        wrap_submodules(
            module,
            l,
            lambda x: _BidirectionalSpectralNormalization(x, n_power_iterations, eps)
        )

