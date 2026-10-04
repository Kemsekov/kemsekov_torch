import torch.nn.init as init
import torch
import torch.nn as nn
from typing import Any

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 conjugate_transpose: bool = False,
                 device: Any | None = None, dtype: Any | None = None):
        super().__init__()
        in_features//=2
        out_features//=2
        # Register parameters properly
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features, 
                                                     device=device, dtype=dtype))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features, 
                                                     device=device, dtype=dtype))
        
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features, 
                                                       device=device, dtype=dtype))
            self.bias_imag = nn.Parameter(torch.randn(out_features, 
                                                       device=device, dtype=dtype))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.in_features = in_features
        self.out_features = out_features
        self.conjugate_transpose = conjugate_transpose
  
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize using complex-aware Xavier/Glorot initialization"""
        # For complex numbers, variance should be split between real and imag parts
        # Standard Xavier: variance = 2 / (fan_in + fan_out)
        # For complex: each part gets half, so variance = 1 / (fan_in + fan_out)
        
        fan_in = self.in_features
        fan_out = self.out_features
        
        # Calculate std for uniform distribution
        std = (1.0 / (fan_in + fan_out)) ** 0.5
        
        # Initialize real and imaginary parts independently
        init.uniform_(self.weight_real, -std, std)
        init.uniform_(self.weight_imag, -std, std)
        
        # Initialize biases to zero
        if self.bias_real is not None:
            init.zeros_(self.bias_real)
            init.zeros_(self.bias_imag)
    
    def forward(self, x):
        real,imag = x.chunk(2,-1)
        
        # Get weight matrices
        W_real = self.weight_real
        W_imag = self.weight_imag
        
        # Apply conjugate transpose if requested
        if self.conjugate_transpose:
            # For conjugate: (a + bi)* = a - bi
            # So we negate the imaginary part of weights
            W_imag = -W_imag
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # Where:
        #   a = W_real, b = W_imag (weights)
        #   c = real, d = imag (input)
        
        # Output real part: W_real @ input_real - W_imag @ input_imag
        out_real = torch.matmul(real, W_real.transpose(-1, -2)) - \
                   torch.matmul(imag, W_imag.transpose(-1, -2))
        
        # Output imaginary part: W_real @ input_imag + W_imag @ input_real
        out_imag = torch.matmul(real, W_imag.transpose(-1, -2)) + \
                   torch.matmul(imag, W_real.transpose(-1, -2))
        
        # Add bias if present
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        
        return torch.concat([out_real, out_imag],-1)
    