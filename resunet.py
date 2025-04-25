from typing import List
from residual import *
from conv_modules import *
from common_modules import Interpolate
class Encoder(torch.nn.Module):
    """
    Encoder module for the Residual U-Net architecture.

    The Encoder progressively downsamples the input tensor using a sequence of ResidualBlocks,
    each followed by an optional attention module (e.g., SCSEModule) and dropout for regularization.
    It preserves intermediate outputs as skip connections, which can be utilized by the Decoder
    for feature fusion during upsampling.

    **Attributes:**
        downs (torch.nn.ModuleList): List of downsampling modules, excluding the final block.
        down5 (torch.nn.Sequential): The final downsampling module in the encoder.
        dropout (torch.nn.Dropout): Dropout layer applied after each downsampling block for regularization.
    """

    def __init__(
        self, 
        in_channels_, 
        out_channels_, 
        dilations, 
        block_sizes,
        dropout_p=0.5,
        attention = SCSEModule,
        dimensions = 2,
        normalization : Literal['batch','instance','group',None] = 'batch',
        kernel_size=4,
        stride = 2,
    ):
        """
        Initializes the Encoder module.

        Constructs a sequence of ResidualBlocks for downsampling, each optionally followed by an attention module
        (e.g., SCSEModule) and dropout. The final ResidualBlock is stored separately as `down5` to mark the end
        of the downsampling path.

        **Args:**
            in_channels_ (List[int]): List of input channel sizes for each ResidualBlock.
            out_channels_ (List[int]): List of output channel sizes for each ResidualBlock.
            dilations (List[List[int]]): List of dilation rates for each ResidualBlock, allowing for multi-scale feature extraction.
            block_sizes (List[int]): Number of convolutional operations (repeats) in each ResidualBlock.
            dropout_p (float, optional): Dropout probability applied after each downsampling block. Default is 0.5.
            attention: Attention module constructor (e.g., SCSEModule) or a list of constructors for each block. If a single constructor is provided, it is applied to all blocks.
            dimensions (int, optional): Dimensionality of the input tensor (1, 2, or 3). Default is 2.
            normalization (Literal['batch','instance',None], optional): Type of normalization to use in ResidualBlocks ('batch', 'instance', or None). Default is 'batch'.
            kernel_size (int,tuple, optional): Kernel size for convolutions in ResidualBlocks. Default is 4.
            stride (int,tuple, optional): what stride to use for convolutions, default is 2

        **Raises:**
            ValueError: If the lengths of `in_channels_`, `out_channels_`, `dilations`, or `block_sizes` do not match.
        """
        super().__init__()
        if not isinstance(attention,list):
            attention=[attention]*len(block_sizes)
        downs_list = []
        for i in range(len(block_sizes)):
            down_i = ResidualBlock(
                in_channels=in_channels_[i],
                out_channels=[out_channels_[i]]*block_sizes[i],
                kernel_size = kernel_size,
                stride = stride,
                dilation=dilations[i],
                normalization=normalization,
                dimensions=dimensions
            )
            attn = attention[i]
            if attn is None:
                attn = torch.nn.Identity()
            else:
                attn = attn(out_channels_[i],dimensions=dimensions)
            down_i = torch.nn.Sequential(down_i,attn)
            downs_list.append(down_i)
        
        self.downs = torch.nn.ModuleList(downs_list[:-1])
        self.down5 = downs_list[-1]
        self.dropout = [nn.Dropout1d,nn.Dropout2d,nn.Dropout3d][dimensions-1](p=dropout_p)
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
            print(x.shape)
            x = down(x)
            x=self.dropout(x)
        x = self.down5(x)
        return x

class Decoder(torch.nn.Module):
    """
    Decoder module for the Residual U-Net architecture.

    The Decoder progressively upsamples the input tensor using a sequence of transposed ResidualBlocks,
    each optionally followed by an attention module (e.g., SCSEModule). It supports concatenation of skip
    connections from the Encoder, reducing the concatenated feature channels using 1x1 convolutions. Dropout
    is applied after each upsampling block to enhance regularization.

    **Attributes:**
        ups (torch.nn.ModuleList): List of upsampling modules, excluding the final block.
        up_1x1_convs (torch.nn.ModuleList): List of 1x1 convolutional layers for reducing channels after skip connection concatenation.
        up5 (ResidualBlock): The final upsampling ResidualBlock in the decoder.
        dropout (torch.nn.Dropout): Dropout layer applied after each upsampling block for regularization.
    """
    def __init__(
        self, 
        up_in_channels, 
        up_out_channels, 
        block_sizes,
        dropout_p=0.5,
        attention = SCSEModule,
        dimensions = 2,
        normalization : Literal['batch','instance','group',None] = 'batch',
        kernel_size=4,
        stride = 2,
    ):
        """
        Initializes the Decoder module.

        Constructs a sequence of transposed ResidualBlocks for upsampling, each optionally followed by an attention module.
        It also sets up 1x1 convolutional layers to reduce the number of channels after concatenating skip connections.
        The final ResidualBlock is stored separately as `up5` to complete the upsampling path.

        **Args:**
            up_in_channels (List[int]): List of input channel sizes for each ResidualBlock in the decoder.
            up_out_channels (List[int]): List of output channel sizes for each ResidualBlock in the decoder.
            block_sizes (List[int]): Number of convolutional operations (repeats) in each ResidualBlock.
            dropout_p (float, optional): Dropout probability applied after each upsampling block. Default is 0.5.
            attention: Attention module constructor (e.g., SCSEModule) or a list of constructors for each block. If a single constructor is provided, it is applied to all blocks.
            dimensions (int, optional): Dimensionality of the input tensor (1, 2, or 3). Default is 2.
            normalization (Literal['batch','instance',None], optional): Type of normalization to use in ResidualBlocks ('batch', 'instance', or None). Default is 'batch'.
            kernel_size (int,tuple, optional): Kernel size for convolutions in ResidualBlocks. Default is 4.
            stride (int,tuple, optional): what stride to use, default is 2

        **Raises:**
            ValueError: If the lengths of `up_in_channels`, `up_out_channels`, or `block_sizes` do not match.
        """
        super().__init__()
        if not isinstance(attention,list):
            attention=[attention]*len(block_sizes)
        ups = []
        conv1x1s = []
        for i in range(len(block_sizes)):
            up_i = ResidualBlock(
                in_channels=up_in_channels[i],
                out_channels=[up_out_channels[i]]*block_sizes[i],
                kernel_size=kernel_size,
                stride = stride,
                dilation=1,
                dimensions=dimensions,
                normalization=normalization,
            ).transpose()
            attn = attention[i]
            if attn is None:
                attn = torch.nn.Identity()
            else:
                attn = attn(up_out_channels[i],dimensions=dimensions)
            up_i = torch.nn.Sequential(up_i,attn)
            
            conv1x1_i = [torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d][dimensions-1](up_out_channels[i]*2, up_out_channels[i], kernel_size=1)
            ups.append(up_i)
            conv1x1s.append(
                nn.Sequential(
                    conv1x1_i,
                    get_normalization_from_name(dimensions,normalization)(up_out_channels[i]))
                )
        self.ups =          torch.nn.ModuleList(ups[:-1])
        self.up_1x1_convs = torch.nn.ModuleList(conv1x1s[:-1])
        self.up5 = ups[-1][0]
        self.dropout = [nn.Dropout1d,nn.Dropout2d,nn.Dropout3d][dimensions-1](p=dropout_p)

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
        # Upsampling path
        for i, (up,conv_1x1) in enumerate(zip(self.ups,self.up_1x1_convs)):
            x = up(x)
            x=self.dropout(x)
            if len(skip_connections)!=0:
                # Concatenate the corresponding skip connection (from the downsampling path)
                skip = skip_connections[-(i + 1)]
                
                # here skip needs to be reshaped to x size before making concat
                x = torch.cat((x, skip), dim=1)
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
        # Upsampling path
        for up in self.ups:
            x = up(x)
            x=self.dropout(x)
        x = self.up5(x)
        return x

class ResidualUnet(torch.nn.Module):
    """
    Residual U-Net architecture combining Encoder and Decoder modules.

    The ResidualUnet forms a U-shaped network by integrating the Encoder and Decoder, utilizing skip
    connections to fuse multi-scale features. It supports customizable output scaling and allows
    fine-tuned configurations for block sizes, attention mechanisms, and normalization strategies.
    The minimum input/output size is 32 due to the downsampling and upsampling operations.

    **Attributes:**
        encoder (Encoder): The Encoder module responsible for the downsampling path.
        decoder (Decoder): The Decoder module responsible for the upsampling path.
        scaler (torch.nn.Module): Module to scale the output tensor relative to the input tensor based on `output_scale`.
        connectors (torch.nn.ModuleList): List of modules to process skip connections before passing them to the Decoder.
    """
    def __init__(
        self,
        in_channels=3, 
        out_channels = 3, 
        block_sizes=[2,2,2,2,2],
        output_scale = 1,
        dimensions=2,
        attention = SCSEModule,
        dropout_p=0.1,
        normalization : Literal['batch','instance','group',None] = 'batch',
        layers_count = 6
        ):
        """
        Initializes the ResidualUnet.

        Constructs the Encoder and Decoder with predefined channel configurations and optional attention mechanisms.
        Sets up an output scaler based on the `output_scale` parameter and initializes connectors to preprocess
        skip connections before decoding.

        **Args:**
            in_channels (int, optional): Number of input channels. Default is 3.
            out_channels (int, optional): Number of output channels. Default is 3.
            block_sizes (List[int], optional): List specifying the number of convolutional operations (repeats) for each ResidualBlock in the Encoder and Decoder. Default is [2, 2, 2, 2, 2].
            output_scale (float, optional): Scaling factor for the output tensor relative to the input size. Must be a positive power of 2. Default is 1 (no scaling).
            dimensions (int, optional): Dimensionality of the input tensor (1, 2, or 3). Default is 2.
            attention: Attention module constructor (e.g., SCSEModule) or a list of constructors for each block in the Encoder and Decoder. If a single constructor is provided, it is applied to all blocks.
            dropout_p (float, optional): Dropout probability applied in the Encoder, Decoder, and connectors. Default is 0.5.
            normalization (Literal['batch','instance','group',None], optional): Type of normalization to use in ResidualBlocks ('batch', 'instance', 'group', or None). Default is 'batch'.
            layers_count: count of layers to use, max is 5, can be smaller if needed

        **Raises:**
            ValueError: If `output_scale` is not a positive power of 2.
        """

        super().__init__()
        assert layers_count<=6,"layers_count must be <= 6"
        # self.input_self_attn = VisualMultiheadSelfAttentionFull(in_channels,in_channels)
        kernel_size=4
        stride = 2
        
        output_scale=float(output_scale)
        channels_ =  [in_channels,32, 64, 128, 256, 512]
        for i in range(1,len(channels_)):
            optimal = channels_[i-1]*stride*dimensions
            if optimal<channels_[i]:
                channels_[i]=optimal
        
        channels_=channels_[:layers_count+1]
        block_sizes=block_sizes[:layers_count]
        
        out_channels_ = channels_[1:]
        in_channels_ = channels_[:-1]
        
        print("in_channels ",in_channels_)
        print("out_channels",out_channels_)
        dilations=[
            1,
            1,
            1,
            1,
            [1]+[3]
        ]
        
        if output_scale==1:
            self.scaler = nn.Identity()

        up_block_sizes = block_sizes[::-1]
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
            block_sizes,
            attention=attention,
            dropout_p=dropout_p,
            dimensions=dimensions,
            normalization=normalization,
            kernel_size=kernel_size,
            stride=stride
        )
        
        self.decoder = Decoder(
            up_in_channels,
            up_out_channels,
            up_block_sizes,
            attention=attention_up,
            dropout_p=dropout_p,
            dimensions=dimensions,
            normalization=normalization,
            kernel_size=kernel_size,
            stride=stride
        )

        self.scaler = Interpolate(scale_factor=output_scale)

        def get_attn(ch):
            if attention is None:
                return nn.Identity()
            return attention(ch,dimensions=dimensions)

        def get_skip_ch(ch):
            skip_ch = ch//4
            if skip_ch<2:
                skip_ch=2
            return skip_ch
        
        # transform that is applied to skip connection before it is passed to decoder
        # out_channels_ = out_channels_[:len(block_sizes)]
        self.connectors = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    in_channels=ch,
                    out_channels=[get_skip_ch(ch),ch],
                    kernel_size=3,
                    normalization=normalization,
                    dimensions=dimensions,
                    dilation=[1]+[2]
                ),
                get_attn(ch),
                [nn.Dropout1d,nn.Dropout2d,nn.Dropout3d][dimensions-1](p=dropout_p),
                # HPB(ch,ch,attn_dropout=dropout_p,ff_dropout=dropout_p),
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
        # x=self.input_self_attn(x)
        x,skip = self.encoder.forward_with_skip(x)
        x=self.scaler(x)

        skip = [torch.jit.fork(t,skip[i]) for i,t in enumerate(self.connectors)]
        skip = [torch.jit.wait(s) for s in skip]
        
        x = self.decoder.forward_with_skip(x,skip)
        
        return x
