import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseBSN(nn.Module):
    def __init__(self, module, n_power_iterations=1, eps=1e-12):
        super().__init__()
        self.module = module
        weight = self.module.weight  # Direct access to weight
        fan_in, fan_out = self._get_fan_in_fan_out(weight)
        
        self.scaling_factor = (2 / fan_in) ** 0.5  # Kaiming scaling
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.has_bias = hasattr(self.module, 'bias')
        self.is_conv = isinstance(self.module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        self.is_conv_t = isinstance(self.module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
        
        # default gamma
        self.gamma = torch.ones(1)
        
        # when possible, make gamma channel-wise to make BSN more flexible
        if self.is_conv or self.is_conv_t:
            # For convolutional layers, gamma is a vector with one value per output channel.
            self.gamma = torch.ones(self.module.out_channels)
            for d in self.module.weight.shape[1:]:
                self.gamma = self.gamma.unsqueeze(-1)
        if self.is_conv_t:
            self.gamma = torch.ones(self.module.out_channels)  # Match output channels
            for d in self.module.weight.shape[2:]:  # Skip in_channels and out_channels
                self.gamma = self.gamma.unsqueeze(-1)
        if hasattr(self.module, 'in_features'):
            # For linear layers, gamma is a vector with one value per input feature.
            self.gamma = torch.ones(self.module.in_features)
        self.gamma = nn.Parameter(self.gamma, requires_grad=True)
        
        self.forward_shape_for_conv_t = [1, 0, *range(2, weight.dim())]
        # Update initialization in __init__
        weight_mat_forward, weight_mat_backward = self.reshape_weight(module.weight)
        u_forward = torch.randn(weight_mat_forward.size(0), device=weight_mat_forward.device, dtype=weight_mat_forward.dtype)  # (out_channels)
        v_forward = torch.randn(weight_mat_forward.size(1), device=weight_mat_forward.device, dtype=weight_mat_forward.dtype)  # (in_channels * k_w * k_h)
        u_backward = torch.randn(weight_mat_backward.size(0), device=weight_mat_backward.device, dtype=weight_mat_backward.dtype)  # (in_channels * k_w * k_h)
        v_backward = torch.randn(weight_mat_backward.size(1), device=weight_mat_backward.device, dtype=weight_mat_backward.dtype)  # (out_channels)
        self.register_buffer("u_forward", u_forward)
        self.register_buffer("v_forward", v_forward)
        self.register_buffer("u_backward",u_backward)
        self.register_buffer("v_backward",v_backward)

    def reshape_weight(self, weight):
        """
        Reshape the weight tensor into two matrices:
        - weight_mat_forward: used for the forward (operator) pass
        - weight_mat_backward: its true transpose, corresponding to the adjoint operator

        For standard convolution layers, the weight tensor has shape
        (out_channels, in_channels, *kernel_dims)
        and we reshape it to (out_channels, in_channels * prod(kernel_dims)).
        
        For conv-transpose layers (shape: (in_channels, out_channels, *kernel_dims)),
        we first permute the first two dimensions so that the forward mapping is from
        in_channels to out_channels and then flatten the kernel dimensions.
        """
        if self.is_conv or self.is_conv_t:
            # For standard convolution layers
            weight_mat_forward = weight.view(weight.size(0), -1)  # (out_channels, in_channels * k1 * k2 * ...)
            weight_mat_backward = weight_mat_forward.t()           # (in_channels * k1 * k2 * ..., out_channels)
        # elif self.is_conv_t:
        #     # For conv-transpose layers, use the stored permutation order (set in __init__)
        #     weight_mat_forward = weight.permute(self.forward_shape_for_conv_t).reshape(weight.size(1), -1)
        #     weight_mat_backward = weight_mat_forward.t()
        else:
            # For linear layers (weight of shape (out_features, in_features))
            weight_mat_forward = weight.view(weight.size(0), -1)
            weight_mat_backward = weight_mat_forward.t()
        return weight_mat_forward, weight_mat_backward
    
    def power_iteration(self, weight):
        """Compute the bidirectional spectral norm using power iteration."""
        weight_mat_forward, weight_mat_backward = self.reshape_weight(weight)

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
        sigma = 0.5*(sigma_forward + sigma_backward)
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

