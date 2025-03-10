from kemsekov_torch.residual import ResidualBlock
from kemsekov_torch.positional_emb import *
from kemsekov_torch.conv_modules import SCSEModule

import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Vec(nn.Module):
    """
    Seq2Vec is a neural network module designed to process input sequences of varying dimensions (1D, 2D, or 3D) and produce a fixed-size output vector. It incorporates positional encoding, residual blocks, squeeze-and-excitation modules, and multi-scale feature extraction to effectively capture both local and global information from the input sequence.

    Attributes:
        encoder (nn.Module): Positional encoding module to inject positional information into the input.
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of channels in the hidden layers.
        dimensions (int): Dimensionality of the input sequence (1 for 1D, 2 for 2D, 3 for 3D).
        compress_input (nn.Sequential): Sequential module to normalize and compress the input.
        conv_layers (nn.Sequential): Sequential module containing residual blocks and squeeze-and-excitation modules.
        multiscale_aspp (nn.ModuleList): Module list for multi-scale atrous spatial pyramid pooling.
        final_layer (nn.Module): Convolutional layer to produce the final output vector.
    """

    def __init__(self, in_channels=256, hidden_channels=512, out_channels=512, dimensions=1, block_sizes=[2, 2, 2, 2, 2, 2], dropout_p=0.1):
        """
        Initializes the SeqToVec module with the specified parameters.

        Args:
            in_channels (int): Number of input channels. Default is 256.
            hidden_channels (int): Number of channels in the hidden layers. Default is 512.
            out_channels (int): Number of output channels. Default is 512.
            dimensions (int): Input spacial dimensions
            block_sizes (list): List specifying the number of residual blocks in each stage. Default is [2, 2, 2, 2, 2].
            dropout_p (float): Dropout probability. Default is 0.1.
        """
        super().__init__()

        # Determine the dimensionality based on the output_size
        self.dimensions = dimensions
        self.hidden_channels=hidden_channels

        # Select appropriate modules based on dimensionality
        dropout = [nn.Dropout1d, nn.Dropout2d, nn.Dropout3d][dimensions - 1]
        avg_pool = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d][dimensions - 1]
        max_pool = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dimensions - 1]
        conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dimensions - 1]
        # bn = nn.SyncBatchNorm
        # bn = [nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d][dimensions-1]
        # Positional encoding module
        self.encoder = PositionalEncodingPermute(in_channels)

        # Input compression module
        self.compress_input = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ZeroPad2d((1, 0, 0, 1)),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_channels),
        )

        # Convolutional layers with residual blocks and squeeze-and-excitation modules
        self.conv_layers = nn.Sequential()
        for i,block_size in enumerate(block_sizes):
            # to downsample input, add some stride
            stride = 1+(i%2)
            layer = nn.Sequential(
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    dimensions=dimensions,
                    conv_impl=[conv] * block_size
                ),
                SCSEModule(hidden_channels, dimensions=dimensions),
                dropout(dropout_p),
            )
            self.conv_layers.append(layer)

        # Multi-scale atrous spatial pyramid pooling (ASPP)
        max_dilation = 4
        assert hidden_channels % max_dilation == 0, f"hidden_channels should be divisible by {max_dilation}"

        chunk = hidden_channels // max_dilation
        dilations = []
        for dilation in [[1 + i] * chunk for i in range(max_dilation)]:
            dilations += dilation

        self.multiscale_aspp = nn.ModuleList()
        kernel_sizes = [3, 3, 3, 3, 3]
        self.min_input_length = 16
        for s in kernel_sizes:
            layer = nn.Sequential(
                max_pool(2, padding=1),
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=s,
                    stride=1,
                    dimensions=dimensions,
                    dilation=dilations,
                    conv_impl=[conv] * 1
                ),
                SCSEModule(hidden_channels, dimensions=dimensions),
                dropout(dropout_p)
            )
            self.multiscale_aspp.append(layer)
        self.aspp_pool = avg_pool(1)
        # Final convolutional layer to produce the output vector
        final_conv_kernel = [1] * dimensions
        final_conv_kernel[-1] = len(kernel_sizes)

        final_conv_stride = [1] * dimensions
        final_conv_stride[-1] = len(kernel_sizes)

        self.final_layer = conv(hidden_channels, out_channels, kernel_size=final_conv_kernel, stride=final_conv_stride)

    def forward(self, x):
        """
        Forward pass of the SeqToVec module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, ...).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, ...).
        """
        # Apply positional encoding
        positional_emb = self.encoder(x)
        if positional_emb is None:
            positional_emb=torch.zeros_like(x)
        
        # Combine input and positional embeddings
        combined = torch.stack([x.flatten(2), positional_emb.flatten(2)], dim=-1)
        
        # Compress the combined input
        compressed_shape = list(x.shape)
        compressed_shape[1] = self.hidden_channels

        compressed = self.compress_input(combined)[:, :, :, 0]
        compressed = compressed.view(compressed_shape)

        # Apply convolutional layers
        transformed = self.conv_layers(compressed)

        # Apply multi-scale ASPP
        aspp_results = [transformed]
        for b in self.multiscale_aspp:
            v = b(aspp_results[-1])
            aspp_results.append(v)
        # for v in aspp_results:
        #     print(v.shape)
        aspp_results = [self.aspp_pool(v) for v in aspp_results]
        aspp_results=torch.concat(aspp_results,dim=-1)
        # Apply final convolutional layer
        result = self.final_layer(aspp_results)
        return result.flatten(1)