import inspect
from typing import Literal
import torch
import torch.nn as nn
from conv_modules import *
from common_modules import *

def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).
    """
    while b:
        a, b = b, a%b
    return a
          
# advanced residual block with modular structure
class ResidualBlock(torch.nn.Module):
    """
    Advanced Residual Block with a Modular Structure.

    This block supports residual convolutional transformations with custom kernel sizes, dilations, 
    and stride. It allows modular selection of the convolution implementation (e.g., BSConvU, standard Conv2d, or ConvTranspose2d)
    and is compatible with transpose convolutions for upsampling.

    The ResidualBlock can be repeated multiple times, with each repeat potentially using a different convolution implementation.
    It also includes an optional batch normalization layer after each convolution and an activation function applied after each normalization.

    Attributes:
        convs (torch.nn.ModuleList): List of convolutional layers applied in sequence for each repeat.
        batch_norms (torch.nn.ModuleList): List of batch normalization layers corresponding to each repeat.
        x_correct (torch.nn.Module): A correction module applied to the input to match the output shape for residual addition.
        activation (torch.nn.Module): The activation function applied after each convolution and normalization.

    Raises:
        AssertionError: 
            - If the length of `conv_impl` (when provided as a list) does not match `repeats`.
            - If the length of `dilation` does not match `kernel_size` when they are provided as lists.
    """

    def __init__(
        self,
        in_channels,                        # Number of input channels.
        out_channels,                       # Number of output channels.
        kernel_size = 3,                    # Kernel size. Could be a list
        stride = 1,                         # stride to use, will reduce/increase output size(dependent on conv2d impl) as multiple of itself
        dilation = 1,                       # List of dilation values for each output channel. Can be an integer
        activation=torch.nn.ReLU,           # Activation function. Always pass constructor
        batch_norm = True,                  #add batch normalization
        conv_impl = nn.Conv2d,              #conv2d implementation. BSConvU torch.nn.Conv2d or torch.nn.ConvTranspose2d or whatever you want
        dimensions : Literal[1,2,3] = 2,
        pad = 0
    ):
        """
        Initializes the ResidualBlock.

        This method sets up the residual block's convolutional layers, batch normalization layers, 
        and residual correction to ensure input-output compatibility for residual addition.

        The residual connection is adjusted using a separate convolution (`x_correct`) if there's a change
        in the number of channels or the stride, ensuring that the input can be added to the output.

        Additionally, the block optimizes computation by collapsing consecutive convolutions with the same
        kernel size and dilation into grouped convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or list of int): Kernel size(s) for the convolutions. If a list is provided, it should match the number of output channels.
            stride (int, optional): Stride for the convolutions, used to modify output spatial dimensions. Default is 1.
            dilation (int or list of int, optional): Dilation rates for the convolutions. If a list is provided, it should match the number of kernel sizes. Default is 1.
            activation (type, optional): Constructor for the activation function to use (e.g., `torch.nn.ReLU`). Default is `torch.nn.ReLU`.
            batch_norm (bool, optional): If True, includes BatchNorm after each convolution. Default is True.
            conv_impl (type or list of type, optional): Convolution implementation class to use. 
                Can be a single type (e.g., `BSConvU`, `nn.Conv2d`, `nn.ConvTranspose2d`) applied to all repeats, 
                or a list of types with length equal to `repeats`, specifying the convolution implementation for each repeat. 
                Default is `BSConvU`.
            pad (int, optional): additional padding for convolutions
        """
        super().__init__()

        if not isinstance(conv_impl,list):
            conv_impl=[conv_impl]
        repeats=len(conv_impl)
        self.added_pad = pad
        self._is_transpose_conv = "output_padding" in inspect.signature(conv_impl[0].__init__).parameters
        self.is_batch_norm = batch_norm
        x_corr_conv_impl = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        x_corr_conv_impl_T = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][dimensions-1]
        if batch_norm:
            # batch_norm_impl=nn.SyncBatchNorm
            batch_norm_impl=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d][dimensions-1]
        else:
            batch_norm_impl=nn.Identity()
        self.dimensions=dimensions
        self._conv_x_correct(in_channels, out_channels, stride, batch_norm_impl, x_corr_conv_impl,x_corr_conv_impl_T)
        
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
        self.conv_impl=conv_impl
        
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

        # collapse same-shaped conv blocks to reduce computation resources
        # for non-first layers
        out_channels_without_dilation = [1]
        kernel_sizes_without_dilation = [kernel_size[0]]
        for c_size in list(kernel_size)[1:]:
            if c_size==kernel_sizes_without_dilation[-1]:
                out_channels_without_dilation[-1]+=1
            else:
                out_channels_without_dilation.append(1)
                kernel_sizes_without_dilation.append(c_size)
        
        self.convs = []
        self.batch_norms = []
        for v in range(repeats):
            # on first repeat block make sure to cast input tensor to output shape
            # and on further repeats just make same-shaped transformations
            in_ch = in_channels if v==0 else out_channels
            stride_ = stride if v==0 else 1
            added_pad = pad if v==0 else 0

            # only at first layer do dilations
            dil = dilations_# if v==0 else [1]*len(out_channels_without_dilation)
            outc = out_channels_# if v==0 else out_channels_without_dilation
            ksizes = kernel_sizes_# if v==0 else kernel_sizes_without_dilation
            
            # Store the conv layers for each output channel with different dilations.
            convs_ = []
            for i in range(len(outc)):
                conv_kwargs = dict(
                    in_channels=in_ch,
                    out_channels=outc[i],
                    kernel_size=ksizes[i],
                    padding=(ksizes[i] + (ksizes[i] - 1) * (dil[i] - 1)) // 2+added_pad,
                    padding_mode="zeros",
                    dilation=dil[i],
                    stride=stride_
                )
                if v==0 and self._is_transpose_conv:
                    conv_kwargs['output_padding']=stride_ - 1
                convs_.append(conv_impl[v](**conv_kwargs))

            conv = torch.nn.ModuleList(convs_)
            self.convs.append(conv)
            
            #optionally add batch normalization
            self.batch_norms.append(batch_norm_impl(out_channels))
            
        self.convs = torch.nn.ModuleList(self.convs)
        self.batch_norms = torch.nn.ModuleList(self.batch_norms)
    def _resize_x_correct(self, in_channels, out_channels, stride, batch_norm_impl, x_corr_conv_impl,x_corr_conv_impl_T):
        scale = 1/stride
        if self._is_transpose_conv:
            scale=stride
        self.x_correct = UpscaleResize(in_channels,out_channels,scale,self.dimensions)
    
    def _conv_x_correct(self, in_channels, out_channels, stride, batch_norm_impl, x_corr_conv_impl,x_corr_conv_impl_T):
        # compute x_size correction convolution arguments so we could do residual addition when we have changed
        # number of channels or some stride
        correct_x_ksize = 1 if stride==1 and self.added_pad==0 else (1+stride)//2 *2 +1
        correct_x_dilation = 1
        correct_x_padding= correct_x_ksize // 2
        
        # make cheap downscale
        x_corr_kwargs=dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = correct_x_ksize,
            dilation=correct_x_dilation,
            stride = stride,
            padding = correct_x_padding+self.added_pad,
            groups=gcd(in_channels,out_channels)
        )
        
        x_conv_impl = x_corr_conv_impl
        if self._is_transpose_conv:
            x_conv_impl = x_corr_conv_impl_T
            x_corr_kwargs['output_padding'] = stride - 1
            x_corr_kwargs['groups'] = 1
        
        # if we have different output tensor size, apply linear x_correction
        # to make sure we can add it with output
        if stride>1 or in_channels!=out_channels or self.added_pad!=0:
            # there is many ways to linearly downsample x, but max pool with conv2d works best of all
            self.x_correct = \
                torch.nn.Sequential(
                    x_conv_impl(**x_corr_kwargs),
                    batch_norm_impl(out_channels)
                )
        else:
            self.x_correct = torch.nn.Identity()
        
    def forward(self, x):
        """
        Applies the residual block transformation to the input tensor.

        This method processes the input through the convolutional transformations, 
        adds the residual correction, and applies the activation function. 
        It leverages TorchScript's parallelism features to optimize performance.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width).
        """
        # Apply each convolution with different dilations to the input and concatenate.
        out_v = x
     
        prev = self.x_correct(x)

        for convs,norm in zip(self.convs,self.batch_norms):
            # Fork to parallelize each convolution operation
            futures = [torch.jit.fork(conv, out_v) for conv in convs]
            # Wait for all operations to complete and collect the results
            results = [torch.jit.wait(future) for future in futures]
            out_v = torch.cat(results, dim=1)
            out_v = self.activation(out_v)+prev
            out_v = norm(out_v)
            prev = out_v
        
        return out_v
    # to make current block work as transpose (which will upscale input tensor) just use different conv2d implementation
    def transpose(self):
        """
        Creates a transposed (upsampling) version of the current ResidualBlock.

        This method returns a new `ResidualBlock` instance configured with
        transposed convolution implementation, enabling the block to perform upsampling operations.

        Returns:
            ResidualBlock: A new `ResidualBlock` instance configured for transposed convolutions.
        """
        
        # if we use stride 1 do not change anything
        conv_impl = [v for v in self.conv_impl]
        conv_impl[0]=[torch.nn.ConvTranspose1d,torch.nn.ConvTranspose2d,torch.nn.ConvTranspose3d][self.dimensions-1]
        
        return ResidualBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size = self.kernel_size,
            stride = self.stride,
            dilation = self.dilation,
            activation = self._activation_func,
            batch_norm = self.is_batch_norm,
            conv_impl = conv_impl,
            dimensions=self.dimensions,
            pad=self.added_pad
        )
