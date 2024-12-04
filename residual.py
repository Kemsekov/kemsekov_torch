import inspect
import torch
import torch.nn as nn

def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).
    """
    while b:
        a, b = b, a%b
    return a

# blueprint pointwise-depthwise convolution, which works better than original depthwise-pointwise convolution
class SEModule(nn.Module):
    """
    Spatial squeeze & channel excitation attention module, as proposed in https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x)
    
class BSConvU(torch.nn.Sequential): 
    """
    Blueprint Pointwise-Depthwise Convolution Block.

    This block implements a convolutional module that performs pointwise convolution
    followed by depthwise convolution. It is an alternative to the standard depthwise-pointwise 
    approach and may be more effective in certain use cases.

    Attributes:
        pw (torch.nn.Conv2d): The pointwise convolution (1x1) layer.
        bn (torch.nn.BatchNorm2d, optional): Optional BatchNorm applied after the pointwise convolution.
        dw (torch.nn.Conv2d): The depthwise convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the depthwise convolution kernel.
        stride (int, optional): Stride for the depthwise convolution. Default is 1.
        padding (int or tuple, optional): Implicit padding for the depthwise convolution. Default is 0.
        dilation (int, optional): Dilation rate for the depthwise convolution. Default is 1.
        bias (bool, optional): Whether to add a learnable bias to the depthwise convolution. Default is True.
        padding_mode (str, optional): Padding mode for the depthwise convolution. Default is 'zeros'.
        with_bn (bool, optional): If True, includes BatchNorm between pointwise and depthwise convolutions. Default is False.
        bn_kwargs (dict, optional): Additional keyword arguments for BatchNorm. Default is None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None): 
        super().__init__() 

        # check arguments 
        if bn_kwargs is None: 
            bn_kwargs = {} 

        # pointwise 
        self.add_module("pw", torch.nn.Conv2d( 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=(1, 1), 
                stride=1, 
                padding=0, 
                dilation=1, 
                groups=1, 
                bias=False, 
        )) 

        # batchnorm 
        if with_bn: 
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs)) 

        # depthwise 
        self.add_module("dw", torch.nn.Conv2d( 
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                groups=out_channels, 
                bias=bias, 
                padding_mode=padding_mode, 
        )) 
          
# advanced residual block with modular structure
class ResidualBlock(torch.nn.Module):
    """
    Advanced Residual Block with Modular Structure.

    This block supports residual convolutional transformations with custom kernel sizes, dilations, 
    and stride. It allows modular selection of the convolution implementation (e.g., BSConvU or 
    standard Conv2d) and is compatible with transpose convolutions for upsampling. 

    Attributes:
        convs (torch.nn.ModuleList): List of convolutional layers applied in sequence.
        batch_norms (torch.nn.ModuleList): List of batch normalization layers corresponding to each repeat.
        x_correct (torch.nn.Module): A linear correction applied to the input to match the output shape for residual addition.
        activation (torch.nn.Module): The activation function applied after each convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or list of int): Kernel size(s) for the convolutions.
        stride (int, optional): Stride for the convolutions, used to modify output spatial dimensions. Default is 1.
        dilation (int or list of int, optional): Dilation rates for the convolutions. Default is 1.
        activation (type, optional): Constructor for the activation function to use. Default is torch.nn.ReLU.
        repeats (int, optional): Number of times to repeat the residual block transformation. Default is 1.
        batch_norm (bool, optional): If True, includes BatchNorm after each convolution. Default is True.
        conv2d_impl (type, optional): Convolution implementation class to use (e.g., BSConvU or Conv2d). Default is BSConvU.
    """

    def __init__(
        self,
        in_channels,                   # Number of input channels.
        out_channels,                  # Number of output channels.
        kernel_size = 3,               # Kernel size. Could be a list
        stride = 1,                    # stride to use, will reduce/increase output size(dependent on conv2d impl) as multiple of itself
        dilation = 1,                  # List of dilation values for each output channel. Can be an integer
        activation=torch.nn.ReLU,      # Activation function. Always pass constructor
        repeats = 1,                   # how many times repeat block internal transformation
        batch_norm = True,             #add batch normalization
        conv2d_impl = BSConvU, #conv2d implementation. BSConvU torch.nn.Conv2d or torch.nn.ConvTranspose2d
    ):
        """
        Initializes the ResidualBlock.

        This method sets up the residual block's convolutional structure, batch normalization, 
        and residual correction to ensure input-output compatibility for residual addition.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or list of int): Kernel size(s) for the convolutions.
            stride (int, optional): Stride for the convolutions, used to modify output spatial dimensions. Default is 1.
            dilation (int or list of int, optional): Dilation rates for the convolutions. Default is 1.
            activation (type, optional): Constructor for the activation function to use. Default is torch.nn.ReLU.
            repeats (int, optional): Number of times to repeat the residual block transformation. Default is 1.
            batch_norm (bool, optional): If True, includes BatchNorm after each convolution. Default is True.
            conv2d_impl (List[type],type, optional): Convolution implementation class to use per each repeat (e.g., BSConvU or Conv2d or Conv2dTranspose). Default is BSConvU

        Raises:
            AssertionError: If the number of dilations does not match the number of output channels.
        """
        super().__init__()

        if not isinstance(conv2d_impl,list):
            conv2d_impl=[conv2d_impl]*repeats

        assert len(conv2d_impl) == repeats, "Length of conv2d_impl must match the number of repeats."
        
        self._is_transpose_conv = "output_padding" in inspect.signature(conv2d_impl[0].__init__).parameters
        
        self.conv_x_correct(in_channels, out_channels, stride, batch_norm, conv2d_impl)
        # self.resize_x_correct(in_channels, out_channels, stride, batch_norm, conv2d_impl)
        
        if not isinstance(dilation,list):
            dilation=[dilation]*out_channels

        if not isinstance(kernel_size,list):
            kernel_size=[kernel_size]*out_channels
        assert len(dilation) == out_channels, "Number of dilations must match the number of output channels."


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self._activation_func = activation
        self.activation = activation()
        self.repeats = repeats
        self.batch_norm = batch_norm
        self.conv2d_impl=conv2d_impl

        # collapse same-shaped conv blocks to reduce computation resources
        out_channels_ = [1]
        kernel_sizes_ = [kernel_size[0]]
        dilations_ = [dilation[0]]
        for c_size,c_dilation in list(zip(kernel_size,dilation))[1:]:
            if c_size==kernel_sizes_[-1] and c_dilation==dilations_[-1]:
                out_channels_[-1]+=1
            else:
                out_channels_.append(1)
                kernel_sizes_.append(c_size)
                dilations_.append(c_dilation)
        
        self.convs = []
        self.batch_norms = []
        for v in range(repeats):
            # on first repeat block make sure to cast input tensor to output shape
            # and on further repeats just make same-shaped transformations
            in_ch = in_channels if v==0 else out_channels
            stride_ = stride if v==0 else 1
            # Store the conv layers for each output channel with different dilations.
            convs_ = []
            for i in range(len(out_channels_)):
                conv_kwargs = dict(
                    in_channels=in_ch,
                    out_channels=out_channels_[i],
                    kernel_size=kernel_sizes_[i],
                    padding=(kernel_sizes_[i] + (kernel_sizes_[i] - 1) * (dilations_[i] - 1)) // 2,
                    padding_mode="zeros",
                    dilation=dilations_[i],
                    stride=stride_
                )
                if v==0 and self._is_transpose_conv:
                    conv_kwargs['output_padding']=stride_ - 1
                convs_.append(conv2d_impl[v](**conv_kwargs))

            conv = torch.nn.ModuleList(convs_)
            self.convs.append(conv)
            
            #optionally add batch normalization
            batch_norm = \
                torch.nn.BatchNorm2d(out_channels) \
                if batch_norm else torch.nn.Identity()
            self.batch_norms.append(batch_norm)
            
        self.convs = torch.nn.ModuleList(self.convs)
        self.batch_norms = torch.nn.ModuleList(self.batch_norms)
        
    def conv_x_correct(self, in_channels, out_channels, stride, batch_norm, conv2d_impl):
        # compute x_size correction convolution arguments so we could do residual addition when we have changed
        # number of channels or some stride
        correct_x_ksize = 1 if stride==1 else (1+stride)//2 *2 +1
        correct_x_dilation = 1
        correct_x_padding= correct_x_ksize // 2
        
        # make cheap downscale
        x_corr_kwargs=dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = correct_x_ksize,
            dilation=correct_x_dilation,
            stride = stride,
            padding = correct_x_padding,
            groups=gcd(in_channels,out_channels)
        )
        x_conv_impl = nn.Conv2d
        if self._is_transpose_conv:
            x_conv_impl = nn.ConvTranspose2d
            x_corr_kwargs['output_padding'] = stride - 1
            x_corr_kwargs['groups'] = 1

        # if we have different output tensor size, apply linear x_correction
        # to make sure we can add it with output
        if stride>1 or in_channels!=out_channels:
            # there is many ways to linearly downsample x, but max pool with conv2d works best of all
            self.x_correct = \
                torch.nn.Sequential(
                    x_conv_impl(**x_corr_kwargs),
                    torch.nn.BatchNorm2d(out_channels) \
                    if batch_norm else torch.nn.Identity()
                )
        else:
            self.x_correct = torch.nn.Identity()

    def forward(self, x):
        """
        Applies the residual block transformation to the input tensor.

        This method processes the input through the convolutional transformations, 
        adds the residual correction, and applies the activation function. 

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width).
        """
        # Apply each convolution with different dilations to the input and concatenate.
        out_v = x
     
        # to speed up parallel computations, make sure to fork all independent convolution computations
        x_corr = torch.jit.fork(self.x_correct, x)

        for convs,norm in zip(self.convs,self.batch_norms):
            # Fork to parallelize each convolution operation
            futures = [torch.jit.fork(conv, out_v) for conv in convs]
            # Wait for all operations to complete and collect the results
            results = [torch.jit.wait(future) for future in futures]
            out_v = torch.cat(results, dim=1)
            out_v = norm(out_v)
            out_v = self.activation(out_v)
         
        x=torch.jit.wait(x_corr)
        # always add x as it was putted-in (without activation applied)
        # so we can get best gradient information flow as x only changed in linear operation (or not changed at all)
        out_v = out_v + x
        return out_v
    # to make current block work as transpose (which will upscale input tensor) just use different conv2d implementation
    def transpose(self,conv2d_transpose_impl = torch.nn.ConvTranspose2d):
        """
        Creates a transposed version of the current ResidualBlock.
        This method returns a new ResidualBlock instance with the specified transpose convolution 
        implementation. The resulting block can be used for upsampling operations.
        Args:
            conv2d_transpose_impl (type, optional): Transpose convolution implementation class to use 
                (e.g., ConvTranspose2d). Default is torch.nn.ConvTranspose2d.
        Returns:
            ResidualBlock: A new ResidualBlock instance configured for transposed convolutions.
        """
        
        # if we use stride 1 do not change anything
        if self.stride==1: 
            conv2d_transpose_impl=self.conv2d_impl
            
        return ResidualBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size = self.kernel_size,
            stride = self.stride,
            dilation = self.dilation,
            activation = self._activation_func,
            repeats = self.repeats,
            batch_norm = self.batch_norm,
            conv2d_impl = conv2d_transpose_impl
        )
