from typing import List
from residual import *
from common_modules import Interpolate

class Encoder(torch.nn.Module):
    """
    Encoder module for the Residual U-Net architecture.

    The Encoder progressively downsamples the input tensor using a sequence of ResidualBlocks,
    each followed by a Squeeze-and-Excitation (SE) module and dropout for regularization.
    It also maintains skip connections that can be used by the Decoder for feature fusion.

    Attributes:
        downs (torch.nn.ModuleList): List of downsampling modules excluding the final block.
        down5 (torch.nn.Sequential): The final downsampling module in the encoder.
        dropout (torch.nn.Dropout2d): Dropout layer applied after each downsampling block.
    """

    def __init__(self, in_channels_, out_channels_, dilations, downs_conv_impl,dropout_p=0.5,attention = SCSEModule):
        """
        Initializes the Encoder module.

        Constructs a sequence of ResidualBlocks, each followed by an attention and dropout.
        The final ResidualBlock is stored separately as `down5`.

        Args:
            in_channels_ (List[int]): List of input channel sizes for each ResidualBlock.
            out_channels_ (List[int]): List of output channel sizes for each ResidualBlock.
            dilations (List[List[int]]): List of dilation rates for each ResidualBlock.
            downs_conv_impl (List[List[type]]): List of convolution implementations for each ResidualBlock.
            dropout_p (float, optional): Dropout probability. Default is 0.5.
            attention: tensor attention implementation

        Raises:
            ValueError: If the lengths of input lists do not match.
        """
        super().__init__()
        downs_list = []
        for i in range(len(downs_conv_impl)):
            down_i = ResidualBlock(
                in_channels=in_channels_[i],
                out_channels=out_channels_[i],
                kernel_size= 3,
                stride = 2,
                dilation=dilations[i],
                conv_impl=downs_conv_impl[i]
            )
            down_i = torch.nn.Sequential(down_i,attention(out_channels_[i]))
            downs_list.append(down_i)
        self.downs =        torch.nn.ModuleList(downs_list[:-1])
        self.down5 = downs_list[-1]
        self.dropout = nn.Dropout2d(p=dropout_p)
    @torch.jit.export
    def forward_with_skip(self, x):
        """
        Forward pass through the Encoder with skip connections.

        This method processes the input tensor through each downsampling block, applies dropout,
        and stores the intermediate outputs as skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - x (torch.Tensor): The output tensor after the final downsampling block.
                - skip_connections (List[torch.Tensor]): List of intermediate tensors for skip connections.
        """
        skip_connections = []
        # Downsampling path
        for down in self.downs:
            x = down(x)
            x=self.dropout(x)
            skip_connections.append(x)
        x = self.down5(x)
        return x, skip_connections
    
    def forward(self,x):
        """
        Forward pass through the Encoder without storing skip connections.

        This method processes the input tensor through each downsampling block and applies dropout,
        but does not retain intermediate outputs for skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output tensor after the final downsampling block.
        """
        for down in self.downs:
            x = down(x)
            x=self.dropout(x)
        x = self.down5(x)
        return x

class Decoder(torch.nn.Module):
    """
    Decoder module for the Residual U-Net architecture.

    The Decoder progressively upsamples the input tensor using a sequence of ResidualBlocks,
    each followed by a Squeeze-and-Excitation (SE) module. It optionally concatenates
    skip connections from the Encoder and reduces the number of feature channels using
    1x1 convolutions. Dropout is applied after each upsampling block for regularization.

    Attributes:
        ups (torch.nn.ModuleList): List of upsampling modules excluding the final block.
        up_1x1_convs (torch.nn.ModuleList): List of 1x1 convolutional layers for channel reduction.
        up5 (ResidualBlock): The final upsampling ResidualBlock.
        dropout (torch.nn.Dropout2d): Dropout layer applied after each upsampling block.
    """
    def __init__(self, up_in_channels, up_out_channels, ups_conv_impl,dropout_p=0.5,attention = SCSEModule):
        """
        Initializes the Decoder module.

        Constructs a sequence of ResidualBlocks configured for upsampling, each followed by an attention.
        Additionally, it sets up 1x1 convolutional layers to reduce the number of channels after concatenation
        with skip connections. The final ResidualBlock is stored separately as `up5`.

        Args:
            up_in_channels (List[int]): List of input channel sizes for each ResidualBlock in the decoder.
            up_out_channels (List[int]): List of output channel sizes for each ResidualBlock in the decoder.
            ups_conv_impl (List[List[type]]): List of convolution implementations for each ResidualBlock.
            dropout_p (float, optional): Dropout probability. Default is 0.5.
            attention: tensor attention implementation

        Raises:
            ValueError: If the lengths of input lists do not match.
        """
        super().__init__()
        ups = []
        conv1x1s = []
        for i in range(len(ups_conv_impl)):
            up_i = ResidualBlock(
                in_channels=up_in_channels[i],
                out_channels=up_out_channels[i],
                kernel_size=3,
                stride = 2,
                dilation=1,
                conv_impl=ups_conv_impl[i]
            )
            up_i = torch.nn.Sequential(up_i,attention(up_out_channels[i]))
            conv1x1_i = torch.nn.Conv2d(up_out_channels[i]*2, up_out_channels[i], kernel_size=1)

            ups.append(up_i)
            conv1x1s.append(conv1x1_i)
        self.ups =          torch.nn.ModuleList(ups[:-1])
        self.up_1x1_convs = torch.nn.ModuleList(conv1x1s[:-1])
        self.up5 = ups[-1][0]
        self.dropout = nn.Dropout2d(p=dropout_p)

    @torch.jit.export
    def forward_with_skip(self,x: torch.Tensor,skip_connections : List[torch.Tensor]):
        """
        Forward pass through the Decoder with skip connections.

        This method processes the input tensor through each upsampling block, optionally concatenates
        corresponding skip connections from the Encoder, applies dropout, and reduces the number of
        feature channels using 1x1 convolutions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            skip_connections (List[torch.Tensor]): List of tensors from the Encoder for skip connections.

        Returns:
            torch.Tensor: The output tensor after the final upsampling block.
        """
        x = self.dropout(x)
        # Upsampling path
        for i, (up,conv_1x1) in enumerate(zip(self.ups,self.up_1x1_convs)):
            x = up(x)
            if len(skip_connections)!=0:
                # Concatenate the corresponding skip connection (from the downsampling path)
                skip = skip_connections[-(i + 1)]
                
                # here skip needs to be reshaped to x size before making concat
                x = torch.cat((x, skip), dim=1)
                x=self.dropout(x)
                # to decrease num of channels
                x = conv_1x1(x)
        x = self.up5(x)
        return x
    

    def forward(self,x: torch.Tensor):
        """
        Forward pass through the Decoder without using skip connections.

        This method processes the input tensor through each upsampling block and applies dropout,
        but does not perform concatenation with skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output tensor after the final upsampling block.
        """
        x = self.dropout(x)
        # Upsampling path
        for up in self.ups:
            x = up(x)
            x=self.dropout(x)
        x = self.up5(x)
        return x
    
class ResidualUnet(torch.nn.Module):
    """
    Residual U-Net architecture combining Encoder and Decoder modules.

    The ResidualUnet integrates the Encoder and Decoder to form a U-shaped network with
    skip connections. It supports flexible output scaling and allows customization of
    block sizes and convolution implementations for both downsampling and upsampling paths.

    Attributes:
        encoder (Encoder): The Encoder module responsible for the downsampling path.
        decoder (Decoder): The Decoder module responsible for the upsampling path.
        scaler (torch.nn.Module): Module to scale the output tensor relative to the input tensor.
    """
    def __init__(self,in_channels=3, out_channels = 3, block_sizes=[2,2,2,2,2],output_scale = 1, attention = SCSEModule):
        """
        Initializes the ResidualUnet.

        Constructs the Encoder and Decoder modules with specified configurations.
        Sets up scaling of the output tensor based on the `output_scale` parameter.

        Args:
            in_channels (int, optional): Number of input channels. Default is 3.
            out_channels (int, optional): Number of output channels. Default is 3.
            block_sizes (List[int], optional): List indicating the number of repeats for each ResidualBlock.
            output_scale (float, optional): Scaling factor for the output tensor. Must be a power of 2.
            attention: tensor attention implementation
            
        Raises:
            ValueError: If `output_scale` is not a positive power of 2.
        """

        super().__init__()
        in_channels_ =  [in_channels,64, 96, 128, 256]
        out_channels_ = [64,         96,128, 256, 512]
        dilations=[
            1,
            1,
            1,
            # aspp block
            [1]*64+[2]*64+[3]*64+[4]*64,
            [1]*256+[2]*128+[3]*128,
        ]
        self.scaler = Interpolate(scale_factor=output_scale)
        
        if output_scale==1:
            self.scaler = nn.Identity()

        downs_conv_impl = [
            [nn.Conv2d]+[BSConvU]*(block_sizes[i]-1) for i in range(len(in_channels_))
        ]

        up_block_sizes = block_sizes[::-1]
        ups_conv_impl = [
            [nn.ConvTranspose2d]+[BSConvU]*(up_block_sizes[i]-1) for i in range(len(in_channels_))
        ]
        up_in_channels = out_channels_[::-1]
        up_out_channels = in_channels_ [::-1]
        up_out_channels[-1]=out_channels
        self.encoder = Encoder(in_channels_,out_channels_,dilations,downs_conv_impl,attention=attention)
        self.decoder = Decoder(up_in_channels,up_out_channels,ups_conv_impl,attention=attention)

    def forward(self, x):
        """
        Forward pass through the Residual U-Net.

        This method processes the input tensor through the Encoder to obtain feature maps and
        skip connections, applies scaling if necessary, and then processes through the Decoder
        using the skip connections to reconstruct the output tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the Residual U-Net.
        """
        x,skip = self.encoder.forward_with_skip(x)
        x=self.scaler(x)
        skip = [self.scaler(i) for i in skip]
        x = self.decoder.forward_with_skip(x,skip)
        return x
