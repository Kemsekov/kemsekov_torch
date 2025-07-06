from typing import Literal
from kemsekov_torch.residual import ResidualBlock
from kemsekov_torch.common_modules import ConcatTensors
from kemsekov_torch.attention import EfficientSpatialChannelAttention
import torch.nn as nn

class ResidualUnet(nn.Module):
    """
    Residual U‑Net architecture with fully dynamic channel configuration
    and support for 1D, 2D, or 3D data.

    This module implements an encoder–decoder (“U‑Net”) style network
    where each downsampling and upsampling block is itself a residual
    block followed by an Efficient Spatial‑Channel Attention layer.

    Parameters
    ----------
    in_channels : int, default=3
        Number of channels in the input tensor.
    out_channels : int, default=3
        Number of channels in the output tensor.
    channels : List[int], default=[64, 128, 256, 256]
        A list of length D defining the number of feature maps at each level.
        - channels[0] is used for the first expansion layer (no downsample).
        - Subsequent entries define the depth of the encoder; longer lists
          produce a deeper UNet.
    dimensions : int in {1,2,3}, default=2
        Number of spatial dimensions. Selects Conv1d/Conv2d/Conv3d accordingly.
    dropout : float, default=0.1
        Dropout probability inside each residual block.
    kernel_size : int or tuple of ints, default=4
        Kernel size for all strided convolutions in the down/up‑sampling
        residual blocks. If tuple, its length must equal `dimensions`.
    stride : int or tuple of ints, default=2
        Stride for all down/up‑sampling convolutions. If tuple, its length
        must equal `dimensions`.
    normalization : {'batch','instance','group','layer', None}, default='group'
        Type of normalization to use inside each residual block.

    Attributes
    ----------
    expand_input : nn.ConvNd
        1×1 convolution projecting `in_channels → channels[0]`.
    collapse_output : nn.ConvNd
        1×1 convolution projecting `channels[0] → out_channels`.
    down_blocks : nn.ModuleList
        Sequence of D depthwise downsampling blocks, each
        `ResidualBlock → Attention`.
    up_blocks : nn.ModuleList
        Sequence of D−1 upsampling blocks (each is a transposed
        residual block → Attention).
    combine_blocks : nn.ModuleList
        Sequence of D−1 “skip‑connection” merges:
        `ConcatTensors → 1×1 conv`.
    final_up : nn.Sequential
        Residual transpose block at the top of the decoder.
    final_combine : nn.Sequential
        Last skip merge with the initial `expand_input` feature map.

    Forward pass
    ------------
    1. Expand input channels: 1×1 conv
    2. For each down block:
       - ResidualBlock(stride, kernel_size)
       - Attention
       - Append to `encodings`
    3. Take the last encoding as the bottleneck.
    4. For each up block + combine:
       - Transposed residual (upsample)
       - Attention
       - Concat with corresponding encoder feature
       - 1×1 conv to fuse
    5. Final upsample and merge with the very first expansion.
    6. Collapse to `out_channels` with a 1×1 conv.

    Examples
    --------
    >>> import torch
    >>> from kemsekov_torch.resunet import ResidualUnet

    # 3D volume UNet, strides=(1,2,2) on depth,height,width
    >>> m3d = ResidualUnet(
    ...     in_channels=3,
    ...     out_channels=3,
    ...     channels=[16, 32, 64],
    ...     dimensions=3,
    ...     kernel_size=(3,4,4),
    ...     stride=(1,2,2)
    ... )
    >>> x3d = torch.randn(8, 3, 40, 128, 128)
    >>> y3d = m3d(x3d)
    >>> y3d.shape
    torch.Size([8, 3, 40, 128, 128])

    # 2D image UNet, square kernels/strides
    >>> m2d = ResidualUnet(
    ...     in_channels=3,
    ...     out_channels=1,
    ...     channels=[32,64,128,256],
    ...     dimensions=2,
    ...     kernel_size=2,
    ...     stride=2
    ... )
    >>> x2d = torch.randn(8, 3, 64, 128)
    >>> y2d = m2d(x2d)
    >>> y2d.shape
    torch.Size([8, 1, 64, 128])

    # 2D UNet with asymmetric downsampling
    >>> m2d_asym = ResidualUnet(
    ...     in_channels=3,
    ...     out_channels=3,
    ...     channels=[32,64,128,256],
    ...     dimensions=2,
    ...     kernel_size=(4,3),
    ...     stride=(2,1)
    ... )
    >>> x2d_asym = torch.randn(8,3,64,128)
    >>> y2d_asym = m2d_asym(x2d_asym)
    >>> y2d_asym.shape
    torch.Size([8,3,64,128])
    """
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        channels=[64, 128, 256, 256],
        dimensions=2,
        dropout=0.1,
        kernel_size=4,
        stride=2,
        normalization: Literal['batch', 'instance', 'group', 'layer', None] = 'group',
    ):
        """
        Unet network to work with multidimensional data (up to 3 dimensions)
        
        Depending on `dimensions` parameters accepts tensors of shape 
            `[batch,channels,d1]` or `[batch,channels,d1,d2]` or `[batch,channels,d1,d2,d3]`
        
        in_channels: input tensor channels
        out_channels: output channels
        channels: ...
        dimensions: input spatial dimensions count
        dropout: dropout probability
        kernel_size: int or list/tuple with dimension-wise kernel_size
        stride: int or list/tuple with dimension-wise stride
        """
        super().__init__()
        assert dimensions>=1 and dimensions<=3, "dimensions parameter should be in range [1:3]"
        if isinstance(kernel_size,list) or isinstance(kernel_size,tuple):
            assert len(kernel_size)==dimensions, f"length of kernel_size should match dimensions parameter, given len({kernel_size})!={dimensions}"
        if isinstance(stride,list) or isinstance(stride,tuple):
            assert len(stride)==dimensions, f"length of stride should match dimensions parameter, given len({stride})!={dimensions}"
        
        # Select convolution type based on dimensions
        conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dimensions - 1]

        self.channels = channels
        self.depth = len(channels)

        # Initial expansion and final collapse
        self.expand_input = conv(in_channels, channels[0], kernel_size=1)
        self.collapse_output = conv(channels[0], out_channels, kernel_size=1)

        common = dict(
            normalization=normalization,
            activation=nn.ReLU,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            dimensions=dimensions
        )

        # Build downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(self.depth):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.down_blocks.append(nn.Sequential(
                ResidualBlock(in_ch, out_ch, **common),
                EfficientSpatialChannelAttention(out_ch)
            ))

        # Build upsampling and combine paths
        self.up_blocks = nn.ModuleList()
        self.combine_blocks = nn.ModuleList()
        for i in reversed(range(1, self.depth)):
            up_block = nn.Sequential(
                ResidualBlock(channels[i], channels[i - 1], **common).transpose(),
                EfficientSpatialChannelAttention(channels[i - 1])
            )
            combine_block = nn.Sequential(
                ConcatTensors(1),
                conv(channels[i - 1] * 2, channels[i - 1], kernel_size=1)
            )
            self.up_blocks.append(up_block)
            self.combine_blocks.append(combine_block)

        # Final upsample to connect to expanded input
        self.final_up = nn.Sequential(
            ResidualBlock(channels[0], channels[0], **common).transpose(),
            EfficientSpatialChannelAttention(channels[0])
        )
        self.final_combine = nn.Sequential(
            ConcatTensors(1),
            conv(channels[0] * 2, channels[0], kernel_size=1)
        )

    def forward(self, x):
        # Store initial expansion for final skip connection
        e = self.expand_input(x)
        x = e
        
        # Down path
        encodings = []
        for down in self.down_blocks:
            print(x.shape)
            x = down(x)
            encodings.append(x)

        # Bottom latent
        x = encodings[-1]
        # Prepare reverse skips excluding bottom
        skips = encodings[:-1][::-1]

        # Up path
        for up, combine, skip in zip(self.up_blocks, self.combine_blocks, skips):
            print(x.shape)
            x = up(x)
            x = combine([x, skip])
        print(x.shape)

        # Final combine with expanded input
        x = self.final_up(x)
        x = self.final_combine([x, e])
        print(x.shape)

        # Collapse to output channels
        return self.collapse_output(x)
