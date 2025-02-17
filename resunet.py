from typing import List
from residual import *
from conv_modules import *
from common_modules import Interpolate
from ittr import HPB
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

    def __init__(self, in_channels_, out_channels_, dilations, downs_conv_impl,dropout_p=0.5,attention = SCSEModule,normalization : Literal['batch','instance',None] = 'batch'):
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
            attention: tensor attention implementation, can be list to define attention layer-wise

        Raises:
            ValueError: If the lengths of input lists do not match.
        """
        super().__init__()
        if not isinstance(attention,list):
            attention=[attention]*len(downs_conv_impl)
        downs_list = []
        for i in range(len(downs_conv_impl)):
            down_i = ResidualBlock(
                in_channels=in_channels_[i],
                out_channels=out_channels_[i],
                kernel_size= 3,
                stride = 2,
                dilation=dilations[i],
                normalization=normalization,
                conv_impl=downs_conv_impl[i]
            )
            down_i = torch.nn.Sequential(down_i,attention[i](out_channels_[i]))
            downs_list.append(down_i)
        
        # at input add batch normalization
        downs_list[0]=torch.nn.Sequential(get_normalization_from_name(2,normalization)(in_channels_[0]),downs_list[0])
        
        self.downs = torch.nn.ModuleList(downs_list[:-1])
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
    def __init__(self, up_in_channels, up_out_channels, ups_conv_impl,dropout_p=0.5,attention = SCSEModule,normalization : Literal['batch','instance',None] = 'batch'):
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
            attention: tensor attention implementation, can be list to define attention layer-wise

        Raises:
            ValueError: If the lengths of input lists do not match.
        """
        super().__init__()
        if not isinstance(attention,list):
            attention=[attention]*len(ups_conv_impl)
        ups = []
        conv1x1s = []
        for i in range(len(ups_conv_impl)):
            up_i = ResidualBlock(
                in_channels=up_in_channels[i],
                out_channels=up_out_channels[i],
                kernel_size=3,
                stride = 2,
                dilation=1,
                normalization=normalization,
                conv_impl=ups_conv_impl[i]
            )
            up_i = torch.nn.Sequential(up_i,attention[i](up_out_channels[i]))
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

    Minimum input/output size is 32.

    The ResidualUnet integrates the Encoder and Decoder to form a U-shaped network with
    skip connections. It supports flexible output scaling and allows customization of
    block sizes and convolution implementations for both downsampling and upsampling paths.

    Attributes:
        encoder (Encoder): The Encoder module responsible for the downsampling path.
        decoder (Decoder): The Decoder module responsible for the upsampling path.
        scaler (torch.nn.Module): Module to scale the output tensor relative to the input tensor.
    """
    def __init__(
        self,
        in_channels=3, 
        out_channels = 3, 
        block_sizes=[2,2,2,2,2],
        output_scale = 1, 
        attention = SCSEModule,
        dropout_p=0.5,
        normalization : Literal['batch','instance',None] = 'batch',
        conv_class_wrapper = lambda x: x
        ):
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
            normalization: what normalization to use when working with data
            
        Raises:
            ValueError: If `output_scale` is not a positive power of 2.
        """

        super().__init__()
        # self.input_self_attn = VisualMultiheadSelfAttentionFull(in_channels,in_channels)
        
        output_scale=float(output_scale)
        in_channels_ =  [in_channels,64, 96, 128, 256]
        out_channels_ = [64,         96,128, 256, 512]
        dilations=[
            1,
            1,
            1,
            # aspp block
            [1]*128+[2]*64+[4]*64,
            [1]*256+[2]*128+[4]*128,
        ]
        
        if output_scale==1:
            self.scaler = nn.Identity()
        
        conv2d = conv_class_wrapper(nn.Conv2d)
        conv2dTranspose = conv_class_wrapper(nn.ConvTranspose2d)
        
        downs_conv_impl = [
            [conv2d]*block_sizes[i] for i in range(len(in_channels_))
        ]

        up_block_sizes = block_sizes[::-1]
        ups_conv_impl = [
            [conv2dTranspose]+[conv2d]*(up_block_sizes[i]-1) for i in range(len(in_channels_))
        ]
        up_in_channels = out_channels_[::-1]
        up_out_channels = in_channels_ [::-1]
        up_out_channels[-1]=out_channels
        
        attention_up=attention
        if isinstance(attention,list):
            attention_up = attention[::-1]
        
        self.encoder = Encoder(
            in_channels_,
            out_channels_,
            dilations,
            downs_conv_impl,
            attention=attention,
            dropout_p=dropout_p,
            normalization=normalization
        )
        
        self.decoder = Decoder(
            up_in_channels,
            up_out_channels,
            ups_conv_impl,
            attention=attention_up,
            dropout_p=dropout_p,
            normalization=normalization
        )

        self.scaler = Interpolate(scale_factor=output_scale)

        # transform that is applied to skip connection before it is passed to decoder
        conv_impl = [conv2d]*(len(out_channels_)-1)
        out_channels_ = out_channels_[:len(conv_impl)]
        self.connectors = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    in_channels=ch,
                    out_channels=ch,
                    kernel_size= 3,
                    stride = 1,
                    conv_impl=conv,
                    normalization=normalization
                ),
                attention(ch),
                nn.Dropout2d(p=dropout_p),
                # HPB(ch,ch,attn_dropout=dropout_p,ff_dropout=dropout_p),
                # scale output
                Interpolate(scale_factor=output_scale)
            ) for conv,ch in zip(conv_impl,out_channels_)
        ])
        
        # add HPB blocks at the end
        for i in [-1,-2]:
            self.connectors[i]=\
                nn.Sequential(
                    HPB(
                        out_channels_[i],
                        out_channels_[i],
                        attn_dropout=dropout_p,
                        ff_dropout=dropout_p,
                        normalization=normalization
                    ),
                    attention(out_channels_[i]),
                    Interpolate(scale_factor=output_scale)
                )
            
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
        # x=self.input_self_attn(x)
        x,skip = self.encoder.forward_with_skip(x)
        x=self.scaler(x)

        skip = [torch.jit.fork(t,skip[i]) for i,t in enumerate(self.connectors)]
        skip = [torch.jit.wait(s) for s in skip]
        
        x = self.decoder.forward_with_skip(x,skip)
        
        return x

class LargeResidualUnet(torch.nn.Module):
    """
    Larger Residual U-Net architecture combining Encoder and Decoder modules.

    Minimum input/output size is 512.

    Attributes:
        encoder (Encoder): The Encoder module responsible for the downsampling path.
        decoder (Decoder): The Decoder module responsible for the upsampling path.
        scaler (torch.nn.Module): Module to scale the output tensor relative to the input tensor.
    """
    def __init__(
        self,
        in_channels=3, 
        out_channels = 3, 
        block_sizes=[2,2,2,2,2,2,2,2],
        output_scale = 1, 
        attention = SCSEModule,
        dropout_p=0.5,
        normalization : Literal['batch','instance',None] = 'batch',
        conv_class_wrapper = lambda x: x):
        """
        Initializes the LargeResidualUnet.

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

        output_scale = float(output_scale)
        in_channels_ =  [in_channels,64, 64, 128,128,192,256,512]
        out_channels_ = [64,         64, 128,128,192,256,512,1024]
        dilations=[
            1,1,1,1,1,
            # aspp block
            [1]*128+[2]*64+[4]*64,
            [1]*256+[2]*128+[4]*128,
            [1]*512+[2]*256+[4]*256,
        ]
        
        if output_scale==1:
            self.scaler = nn.Identity()
        conv2d = conv_class_wrapper(nn.Conv2d)
        conv2dTranspose = conv_class_wrapper(nn.ConvTranspose2d)
        
        downs_conv_impl = [
            [conv2d]*block_sizes[i] for i in range(len(in_channels_))
        ]

        up_block_sizes = block_sizes[::-1]
        ups_conv_impl = [
            [conv2dTranspose]+[conv2d]*(up_block_sizes[i]-1) for i in range(len(in_channels_))
        ]
        up_in_channels = out_channels_[::-1]
        up_out_channels = in_channels_ [::-1]
        up_out_channels[-1]=out_channels
        
        attention_up=attention
        if isinstance(attention,list):
            attention_up = attention[::-1]
        
        self.encoder = Encoder(
            in_channels_,
            out_channels_,
            dilations,
            downs_conv_impl,
            attention=attention,
            dropout_p=dropout_p,
            normalization=normalization
        )
        self.decoder = Decoder(
            up_in_channels,
            up_out_channels,
            ups_conv_impl,
            attention=attention_up,
            dropout_p=dropout_p,
            normalization=normalization
        )

        self.scaler = Interpolate(scale_factor=output_scale)

        # transform that is applied to skip connection before it is passed to decoder
        self.connectors = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    in_channels=ch,
                    out_channels=ch,
                    kernel_size= 3,
                    stride = 1,
                    conv_impl=conv2d,
                    normalization=normalization
                ),
                attention(ch),
                nn.Dropout2d(p=dropout_p),
                # scale output
                Interpolate(scale_factor=output_scale)
            ) for ch in out_channels_[:-1]
        ])

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

        skip = [torch.jit.fork(t,skip[i]) for i,t in enumerate(self.connectors)]
        
        skip = [torch.jit.wait(s) for s in skip]
        
        x = self.decoder.forward_with_skip(x,skip)
        return x
