import torch
from torch.nn import Parameter
from torch.nn.functional import normalize
import torch
from torch.nn import Parameter
from torch.nn.functional import normalize

class BidirectionalSpectralNorm:
    """
    Applies bidirectional spectral normalization to a given module's weight parameter.
    This implementation normalizes the weight matrix during both forward and backward passes.
    """

    def __init__(self, module, name='weight', n_power_iterations=1, eps=1e-12):
        """
        Initializes the BidirectionalSpectralNorm.

        Args:
            module (nn.Module): The module to which spectral normalization is applied.
            name (str): The name of the weight parameter to normalize.
            n_power_iterations (int): Number of power iterations for spectral norm approximation.
            eps (float): Small value to prevent division by zero during normalization.
        """
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        if not self._has_params():
            self._initialize_params()
        self._register_backward_hook()

    def _initialize_params(self):
        """
        Initializes the parameters required for spectral normalization.
        """
        weight = getattr(self.module, self.name)
        height = weight.size(0)
        width = weight.view(height, -1).size(1)

        u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=self.eps)
        v = normalize(weight.new_empty(width).normal_(0, 1), dim=0, eps=self.eps)

        self.module.register_parameter(self.name + '_orig', Parameter(weight.data))
        self.module.register_buffer(self.name + '_u', u)
        self.module.register_buffer(self.name + '_v', v)

    def _has_params(self):
        """
        Checks if the necessary parameters for spectral normalization exist.

        Returns:
            bool: True if parameters exist, False otherwise.
        """
        return hasattr(self.module, self.name + '_u') and hasattr(self.module, self.name + '_v')

    def _update_u_v(self):
        """
        Updates the 'u' and 'v' vectors used for spectral normalization.
        """
        weight = getattr(self.module, self.name + '_orig')
        height = weight.size(0)
        weight_mat = weight.view(height, -1)

        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')

        for _ in range(self.n_power_iterations):
            v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
            u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight_sn = weight / sigma
        setattr(self.module, self.name, weight_sn)

    def _register_backward_hook(self):
        """
        Registers a backward hook to adjust gradients during backpropagation.
        """
        def backward_hook(module, grad_input, grad_output):
            if not self.module.training:
                return grad_input  # No modification during evaluation

            weight = getattr(module, self.name + '_orig')
            height = weight.size(0)
            weight_mat = weight.view(height, -1)

            u = getattr(module, self.name + '_u')
            v = getattr(module, self.name + '_v')

            for _ in range(self.n_power_iterations):
                v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)

            sigma = torch.dot(u, torch.matmul(weight_mat, v))
            print(grad_input)
            grad_input = tuple(g / sigma if g is not None else None for g in grad_input)
            return grad_input

        self.module.register_full_backward_hook(backward_hook)

    def __call__(self, *args):
        """
        Updates the spectral normalization parameters and performs the forward pass.

        Args:
            *args: Arguments to pass to the module's forward method.

        Returns:
            The output of the module's forward pass.
        """
        if self.module.training:
            self._update_u_v()
        return self.module(*args)

def apply_bidirectional_spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12):
    BidirectionalSpectralNorm(module, name, n_power_iterations, eps)
    return module