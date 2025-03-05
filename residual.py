import inspect
from typing import Literal
import torch
import torch.nn as nn
from conv_modules import *
from common_modules import *

# advanced residual block with modular structure
class ResidualBlock(torch.nn.Module):
    """
    ResidualBlock

    Advanced Residual Block with a Modular and Flexible Structure.

    This module implements a highly configurable residual block that applies repeated convolutional transformations with optional parameter collapsing and parallel execution. Each block repeat can use a different convolution implementation (e.g., standard Conv2d, BSConvU, or ConvTranspose2d) to perform the transformation, and supports custom kernel sizes, dilations, strides, and padding. The block ensures compatibility for residual addition by adjusting the input either via a dedicated convolution or through interpolation-based resizing. It is designed to support both downsampling and upsampling operations.

    Parameters:
        in_channels (int):
            Number of channels in the input tensor.
        out_channels (int or list of int):
            Number of output channels for each block repeat. If a single integer is provided, it is converted internally to a list.
        kernel_size (int or list of int):
            Kernel size(s) for the convolutional layers. When provided as a list, its length should match the total number of convolution operations in the first repeat. Consecutive operations with identical kernel size and dilation may be collapsed to reduce computation.
        stride (int, optional):
            Stride for the convolution in the first block repeat. Subsequent repeats use a stride of 1. This parameter controls spatial dimension changes.
        dilation (int or list of int, optional):
            Dilation rate(s) for the convolutional layers. If provided as a list, its length should match that of kernel_size for the first block.
        activation (callable, optional):
            Constructor for the activation function (e.g., torch.nn.ReLU). An instance is created during initialization. Default is torch.nn.ReLU.
        normalization (Literal['batch', 'instance', None], optional):
            Specifies the type of normalization to apply after convolution. Options include 'batch', 'instance', or None. Default is 'batch'.
        conv_impl (type or list of type, optional):
            Convolution implementation(s) to use. This can be a single convolution type (e.g., nn.Conv2d, BSConvU, or nn.ConvTranspose2d) applied to all block repeats, or a list specifying different implementations for each repeat.
        dimensions (Literal[1, 2, 3], optional):
            Dimensionality of the convolution (1D, 2D, or 3D). Default is 2.
        pad (int, optional):
            Additional padding to apply to the convolutional layers.
        x_residual_type (Literal['conv', 'resize', `None`], optional):
            Method for adjusting the input tensor to match the output shape for residual addition.
            - 'conv': Applies a dedicated convolutional correction (with optional normalization) to align the input.
            - 'resize': Uses interpolation-based resizing (nearest neighbor) for spatial adjustment (requires pad to be 0).
            - `None`: Removes residual connection from module at all

    Attributes:
        convs (ModuleList):
            A ModuleList containing sub-ModuleLists of convolutional layers for each block repeat. Convolutions with identical parameters may be grouped and executed in parallel using torch.jit.fork.
        norms (ModuleList):
            A list of normalization layers, one per block repeat.
        x_correct (Module):
            A correction module that adjusts the input tensor (via convolution or resizing) to be compatible with the output tensor for residual addition.
        activation (Module):
            The instantiated activation function applied after each normalization.
        repeats (int):
            The number of repeated convolutional transformations (i.e., the number of block repeats).
        _is_transpose_conv (bool):
            Internal flag indicating whether the convolution implementation supports transposed convolutions (i.e., upsampling), determined by the presence of the 'output_padding' parameter in the convolution’s constructor.

    Methods:
        forward(x: Tensor) -> Tensor:
            Executes the forward pass by applying the sequence of convolutional layers (leveraging parallel execution), followed by normalization and activation, and finally adds the corrected residual input.
        transpose() -> ResidualBlock:
            Returns a new ResidualBlock instance configured for transposed convolutions, enabling upsampling. In the transposed version, the first block’s convolution implementation is replaced with the corresponding transposed convolution type while subsequent blocks remain unchanged.

    Usage Examples:
        >>> m = ResidualBlock(
        ...     24,
        ...     [128, 64],
        ...     kernel_size=[3]*4 + [4]*4 + [5]*8,
        ...     dilation=[1]*8 + [2]*4 + [3]*4,
        ...     stride=2,
        ...     pad=0
        ... )
        >>> output = m(torch.randn(2, 24, 32, 64))
        >>> output.shape
        torch.Size([2, 64, 16, 32])
        >>> m_transposed = m.transpose()
        >>> output_transposed = m_transposed(torch.randn(2, 24, 32, 64))
        >>> output_transposed.shape
        torch.Size([2, 64, 64, 128])

    Notes:
        - The module collapses consecutive convolutional operations with the same kernel size (and dilation for the first block) to reduce computational overhead.
        - When using transposed convolutions via the transpose() method, the block is reconfigured to perform upsampling while preserving the residual structure.
        - If using 'resize' for the residual path, ensure that pad is set to 0.
    """


    def __init__(
        self,
        in_channels,                        # Number of input channels.
        out_channels,                       # Number of output channels.
        kernel_size = 3,                    # Kernel size. Could be a list
        stride = 1,                         # stride to use, will reduce/increase output size(dependent on conv2d impl) as multiple of itself
        dilation = 1,                       # List of dilation values for each output channel. Can be an integer
        activation=torch.nn.ReLU,           # Activation function. Always pass constructor
        normalization : Literal['batch','instance','group',None] = 'batch',#which normalization to use
        dimensions : Literal[1,2,3] = 2,
        pad = 0,
        conv_impl = None,              #conv2d implementation. BSConvU torch.nn.Conv2d or torch.nn.ConvTranspose2d or whatever you want
        x_residual_type : Literal['conv','resize'] = 'resize',
        padding_mode="zeros"
    ):
        """
        ResidualBlock

        Advanced Residual Block with a Modular and Flexible Structure.

        This module implements a highly configurable residual block that applies repeated convolutional transformations with optional parameter collapsing and parallel execution. Each block repeat can use a different convolution implementation (e.g., standard Conv2d, BSConvU, or ConvTranspose2d) to perform the transformation, and supports custom kernel sizes, dilations, strides, and padding. The block ensures compatibility for residual addition by adjusting the input either via a dedicated convolution or through interpolation-based resizing. It is designed to support both downsampling and upsampling operations.

        Parameters:
            in_channels (int):
                Number of channels in the input tensor.
            out_channels (int or list of int):
                Number of output channels for each block repeat. If a single integer is provided, it is converted internally to a list.
            kernel_size (int or list of int):
                Kernel size(s) for the convolutional layers. When provided as a list, its length should match the total number of convolution operations in the first repeat. Consecutive operations with identical kernel size and dilation may be collapsed to reduce computation.
            stride (int, optional):
                Stride for the convolution in the first block repeat. Subsequent repeats use a stride of 1. This parameter controls spatial dimension changes.
            dilation (int or list of int, optional):
                Dilation rate(s) for the convolutional layers. If provided as a list, its length should match that of kernel_size for the first block.
            activation (callable, optional):
                Constructor for the activation function (e.g., torch.nn.ReLU). An instance is created during initialization. Default is torch.nn.ReLU.
            normalization (Literal['batch', 'instance','group', None], optional):
                Specifies the type of normalization to apply after convolution. Options include 'batch', 'instance', or None. Default is 'batch'.
            dimensions (Literal[1, 2, 3], optional):
                Dimensionality of the convolution (1D, 2D, or 3D). Default is 2.
            pad (int, optional):
                Additional padding to apply to the convolutional layers.
            conv_impl (type or list of type, optional):
                Convolution implementation(s) to use. This can be a single convolution type (e.g., nn.Conv2d, BSConvU, or nn.ConvTranspose2d) applied to all block repeats, or a list specifying different implementations for each repeat.
            x_residual_type (Literal['conv', 'resize', 'None'], optional):
                Method for adjusting the input tensor to match the output shape for residual addition.
                - 'conv': Applies a dedicated convolutional correction (with optional normalization) to align the input.
                - 'resize': Uses interpolation-based resizing (nearest neighbor) for spatial adjustment (requires pad to be 0).
                - `None`: Removes residual connection from module
            padding_mode: how to pad convolutions. One of ['constant', 'reflect', 'replicate' or 'circular']

        Attributes:
            convs (ModuleList):
                A ModuleList containing sub-ModuleLists of convolutional layers for each block repeat. Convolutions with identical parameters may be grouped and executed in parallel using torch.jit.fork.
            norms (ModuleList):
                A list of normalization layers, one per block repeat.
            x_correct (Module):
                A correction module that adjusts the input tensor (via convolution or resizing) to be compatible with the output tensor for residual addition.
            activation (Module):
                The instantiated activation function applied after each normalization.
            repeats (int):
                The number of repeated convolutional transformations (i.e., the number of block repeats).
            _is_transpose_conv (bool):
                Internal flag indicating whether the convolution implementation supports transposed convolutions (i.e., upsampling), determined by the presence of the 'output_padding' parameter in the convolution’s constructor.

        Methods:
            forward(x: Tensor) -> Tensor:
                Executes the forward pass by applying the sequence of convolutional layers (leveraging parallel execution), followed by normalization and activation, and finally adds the corrected residual input.
            transpose() -> ResidualBlock:
                Returns a new ResidualBlock instance configured for transposed convolutions, enabling upsampling. In the transposed version, the first block’s convolution implementation is replaced with the corresponding transposed convolution type while subsequent blocks remain unchanged.

        Usage Examples:
            >>> m = ResidualBlock(
            ...     24,
            ...     [128, 64],
            ...     kernel_size=[3]*4 + [4]*4 + [5]*8,
            ...     dilation=[1]*8 + [2]*4 + [3]*4,
            ...     stride=2,
            ...     pad=0
            ... )
            >>> output = m(torch.randn(2, 24, 32, 64))
            >>> output.shape
            torch.Size([2, 64, 16, 32])
            >>> m_transposed = m.transpose()
            >>> output_transposed = m_transposed(torch.randn(2, 24, 32, 64))
            >>> output_transposed.shape
            torch.Size([2, 64, 64, 128])

        Notes:
            - The module collapses consecutive convolutional operations with the same kernel size (and dilation for the first block) to reduce computational overhead.
            - When using transposed convolutions via the transpose() method, the block is reconfigured to perform upsampling while preserving the residual structure.
            - If using 'resize' for the residual path, ensure that pad is set to 0.
        """

        super().__init__()
        
        x_corr_conv_impl = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        x_corr_conv_impl_T = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][dimensions-1]
        
        if conv_impl is None:
            conv_impl=x_corr_conv_impl
        
        if x_residual_type=='resize':
            assert pad==0, "when using 'resize' x_residual_type pad must be zero"
            
        if not isinstance(conv_impl,list):
            conv_impl=[conv_impl]
        
        if not isinstance(out_channels,list):
            out_channels=[out_channels]
        if len(conv_impl)==1:
            conv_impl = conv_impl*len(out_channels)
        
        assert len(out_channels)==len(conv_impl),f"len(out_channels) must equal len(conv_impl), {len(out_channels)}!={len(conv_impl)}"
                
        repeats=len(conv_impl)

        self.added_pad = pad
        self.padding_mode=padding_mode
        self._is_transpose_conv = "output_padding" in inspect.signature(conv_impl[0].__init__).parameters
        if self._is_transpose_conv:
            assert pad==0, f"transpose ResidualBlock works only with pad=0, given pad {pad}!=0"

        self.normalization = normalization
        if not isinstance(kernel_size,list):
            kernel_size=[kernel_size]*out_channels[0]
        # assert all([v%2==1 for v in kernel_size]), f"kernel size must be odd number, but given kernel size {kernel_size}"
        self.kernel_size = kernel_size
        
        
        norm_impl = get_normalization_from_name(dimensions,normalization)
        
        self.dimensions=dimensions
        if x_residual_type == 'conv':
            self._conv_x_correct(in_channels, out_channels[-1], stride, norm_impl, x_corr_conv_impl,x_corr_conv_impl_T)
        
        if x_residual_type == 'resize':
            self._resize_x_correct(in_channels, out_channels[-1], stride, norm_impl, x_corr_conv_impl,x_corr_conv_impl_T)
        if x_residual_type is None:
            self._no_x_residual()
            
        
        assert x_residual_type in ["conv","resize", None],"x_residual_type must be one of ['conv','resize',None], but got "+x_residual_type
        self.x_residual_type=x_residual_type
        if not isinstance(dilation,list):
            dilation=[dilation]*out_channels[0]

        # assert len(dilation) == out_channels[0], f"Number of dilations must match the number of output channels at first dim. {len(dilation)} != {out_channels[0]}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self._activation_func = activation
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
        assert len(kernel_size)==len(dilation), f"len(kernel_size) must equal len(dilation), {len(kernel_size)} != {len(dilation)}"
        
        self.convs = []
        self.norms = []
        for v in range(repeats):
            # on first repeat block make sure to cast input tensor to output shape
            # and on further repeats just make same-shaped transformations
            in_ch = in_channels if v==0 else out_channels[v-1]
            stride_ = stride if v==0 else 1
            added_pad = pad if v==0 else 0

            # do not change
            dil = dilations_ if v==0 else [1]*len(out_channels_without_dilation)
            
            # change to match out_channels[v]
            outc = out_channels_ if v==0 else out_channels_without_dilation
            outc = [int(c/sum(outc)*out_channels[v]) for c in outc]
            
            # channels that is remained to be added
            remaining_channels = out_channels[v]-sum(outc)
            
            # do not change
            ksizes = kernel_sizes_ if v==0 else kernel_sizes_without_dilation
            
            outc[torch.argmin(torch.tensor(ksizes))]+=remaining_channels
            
            # Store the conv layers for each output channel with different dilations.
            convs_ = []
            for i in range(len(outc)):
                ks = ksizes[i]
                # actual conv kernel size with dilation

                if ks%2==0:
                    if v==0:
                        assert stride_!=1,f"Impossible to use kernel_size={ks} with stride=1 to get same-shaped tensor"
                    if stride_==1:
                        ks+=1
                ks_with_dilation = ks + (ks - 1) * (dil[i] - 1)
                
                if ks_with_dilation%2==0:
                    compensation = -1
                else:
                    compensation=0
                
                conv_kwargs = dict(
                    in_channels=in_ch,
                    out_channels=outc[i],
                    kernel_size=ks,
                    padding = ks_with_dilation // 2 + added_pad + compensation,
                    dilation=dil[i],
                    stride=stride_,
                    padding_mode=padding_mode
                )
                if v==0 and self._is_transpose_conv:
                    conv_kwargs['output_padding']=stride_ - 1 + added_pad + compensation
                convs_.append(conv_impl[v](**conv_kwargs))

            conv = torch.nn.ModuleList(convs_)
            self.convs.append(conv)
            
            #optionally add normalization
            self.norms.append(norm_impl(out_channels[v]))
            
        self.convs = torch.nn.ModuleList(self.convs)
        self.norms = torch.nn.ModuleList(self.norms)
        
        # re-zero
        if x_residual_type is None:
            self.alpha = 1
        else:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        
        def create_activation(activation_class):
            # Get the constructor signature
            signature = inspect.signature(activation_class.__init__)
            
            # Check if 'inplace' is in the parameters
            if 'inplace' in signature.parameters:
                return activation_class(inplace=True)
            else:
                return activation_class()
        # for each layer create it's own activation function
        self.activation = nn.ModuleList([create_activation(activation) for i in self.convs])
    
    def _no_x_residual(self):
        self.x_correct = ConstModule()
        
    def _resize_x_correct(self, in_channels, out_channels, stride, norm_impl, x_corr_conv_impl,x_corr_conv_impl_T):
        scale = 1/stride
        if self._is_transpose_conv:
            scale=stride
        self.x_correct = UpscaleResize(in_channels,out_channels,scale,self.dimensions,normalization=self.normalization,mode='nearest-exact')
    
    def _conv_x_correct(self, in_channels, out_channels, stride, norm_impl, x_corr_conv_impl,x_corr_conv_impl_T):
        # compute x_size correction convolution arguments so we could do residual addition when we have changed
        # number of channels or some stride
        correct_x_ksize = 1 if stride==1 and self.added_pad==0 else min(self.kernel_size)
        correct_x_padding= correct_x_ksize // 2 + self.added_pad
        if correct_x_ksize%2==0:
            compensation = -1
        else:
            compensation = 0
            
        correct_x_dilation = 1
        
        # make cheap downscale
        x_corr_kwargs=dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = correct_x_ksize,
            dilation=correct_x_dilation,
            stride = stride,
            padding = correct_x_padding+compensation,
            padding_mode=self.padding_mode
            # groups=gcd(in_channels,out_channels)
        )
        
        x_conv_impl = x_corr_conv_impl
        if self._is_transpose_conv:
            x_conv_impl = x_corr_conv_impl_T
            x_corr_kwargs['output_padding'] = stride - 1 + self.added_pad+compensation
            x_corr_kwargs['groups'] = 1
        
        # if we have different output tensor size, apply linear x_correction
        # to make sure we can add it with output
        if stride>1 or in_channels!=out_channels or self.added_pad!=0:
            # there is many ways to linearly downsample x, but max pool with conv2d works best of all
            self.x_correct = \
                torch.nn.Sequential(
                    x_conv_impl(**x_corr_kwargs),
                    norm_impl(out_channels)
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
        x_corr = self.x_correct(x)
        out_v = x
        
        for convs,norm,act in zip(self.convs,self.norms,self.activation):
            # Fork to parallelize each convolution operation
            futures = [torch.jit.fork(conv, out_v) for conv in convs]
            # Wait for all operations to complete and collect the results
            results = [torch.jit.wait(future) for future in futures]
            out_v = torch.cat(results, dim=1)
            out_v = act(norm(out_v))
        
        return self.alpha*out_v+x_corr
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
        
        if self.stride!=1:
            conv_impl[0]=[torch.nn.ConvTranspose1d,torch.nn.ConvTranspose2d,torch.nn.ConvTranspose3d][self.dimensions-1]
        
        return ResidualBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size = self.kernel_size,
            stride = self.stride,
            dilation = self.dilation,
            activation = self._activation_func,
            normalization=self.normalization,
            conv_impl = conv_impl,
            dimensions=self.dimensions,
            pad=self.added_pad,
            x_residual_type=self.x_residual_type,
            padding_mode="zeros"
        )