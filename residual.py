import inspect
import math
from typing import Literal
import torch
import torch.nn as nn
from conv_modules import *
from common_modules import *
class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        dilation = 1,
        dropout = 0.0,
        activation=torch.nn.SiLU,
        normalization : Literal['batch','instance','group','spectral','layer',None] = None,
        dimensions : Literal[1,2,3] = 2,
        is_transpose = False,
        padding_mode : Literal['zeros','constant', 'reflect', 'replicate', 'circular']="zeros",
        device = None,
        disable_residual = False
    ):
        """
        Creates general-use residual block.
        
        * in_channels: size of input channels
        * out_channels: expected output channels. It can be integer or a list, defining a chain of convolutions, like [16,32,8], which will produce three internal convolutions `input_channels -> 16 -> 32-> 8`
        * kernel_size: integer, or tuple/list with dimensions-wise kernel size for convolutions.
        * stride: integer, or tuple/list with dimensions-wise stride for convolutions.
        * dilation: List that defines required dilations. Applied only to first convolution. Example: `[1]+[2]+[4]*3` will do 1/5 convolutions with dilation 1, 1/5 with dilation 2 and 3/5 with dilation 4
        * dropout: dropout probability
        * activation: activation function
        * normalization: one of `['batch','instance','group','spectral','layer',None]`, applies required normalization. Defaults to `None` because of using re-zero approach to using skip connection. `'group'` will use hubristic to determine optimal number of groups.
        * dimensions: input tensor dimensions, selects one of conv1d conv2d conv3d implementation for convolutions
        * is_transpose: do we need to use transpose convolutions. I advice you to use method `ResidualBlock.transpose(self)` instead of setting this argument manually
        * padding_mode: what padding to use in convolutions, by default will use `'replicate'` when possible
        * device: where to put weights of intialized module
        * disable_residual: converts given block to non-resiual
        """
        super().__init__()
        
        self.__init_device = device
        
        self.dropout = [nn.Dropout1d,nn.Dropout2d,nn.Dropout3d][dimensions-1](dropout)
        if dropout==0:
            dropout_impl=nn.Identity()
        
        x_corr_conv_impl = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        x_corr_conv_impl_T = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][dimensions-1]
            
        if not (isinstance(out_channels,list) or isinstance(out_channels,tuple)):
            out_channels=[out_channels]
        
        repeats=len(out_channels)
        self.padding_mode=padding_mode
        self._is_transpose_conv = is_transpose
        self._dropout_p = dropout

        self.normalization = normalization
        if not (isinstance(kernel_size,list) or isinstance(kernel_size,tuple)):
            kernel_size=[kernel_size]*dimensions
        if not (isinstance(stride,list) or isinstance(stride,tuple)):
            stride = [stride]*dimensions

        self.kernel_size = kernel_size

        # handle spectral normalization
        if normalization=='spectral':
            norm_impl=lambda *x: nn.Identity()
            old_x_corr = x_corr_conv_impl
            x_corr_conv_impl=lambda *x,**y: nn.utils.spectral_norm(old_x_corr(*x,**y))
            old_x_corr_T = x_corr_conv_impl_T
            x_corr_conv_impl_T=lambda *x,**y: nn.utils.spectral_norm(old_x_corr_T(*x,**y))
        else:    
            norm_impl = get_normalization_from_name(dimensions,normalization)
        self.dimensions=dimensions
        self._conv_x_linear(in_channels, out_channels[-1], stride, norm_impl, x_corr_conv_impl,x_corr_conv_impl_T,device)
        
        if not isinstance(dilation,list):
            dilation=[dilation]*out_channels[0]
        assert len(kernel_size)==dimensions,f'kernel_size length expected to have {dimensions} elements, found {kernel_size}'
        assert len(stride)==dimensions,f'stride length expected to have {dimensions} elements, found {stride}'
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self._activation_func = activation
        self.repeats = repeats
        
        # collapse same-shaped conv blocks to reduce computation resources
        out_channels_ = [1]
        dilations_ = [dilation[0]]
        
        for c_dilation in (dilation)[1:]:
            if c_dilation==dilations_[-1]:
                out_channels_[-1]+=1
            else:
                out_channels_.append(1)
                dilations_.append(c_dilation)
        # collapse same-shaped conv blocks to reduce computation resources
        
        self.convs = []
        self.norms = []
        
        assert all([s%2==0 or s==1 for s in stride]), f"all stride must be even or 1. given stride={stride}"
        
        strides = torch.tensor([[1]*dimensions]*repeats)
        
        for d in range(dimensions):
            stride_ind=0
            while math.prod([v[d] for v in strides])<stride[d]:
                strides[stride_ind%repeats][d]*=2
                stride_ind+=1
        
        for v in range(repeats):
            # on first repeat block make sure to cast input tensor to output shape
            # and on further repeats just make same-shaped transformations
            in_ch = in_channels if v==0 else out_channels[v-1]
            stride_ = strides[v]

            # do not change
            dil = dilations_ if v==0 else [1]*len(dilations_)
            
            # change to match out_channels[v]
            outc = out_channels_
            outc = [int(c/sum(outc)*out_channels[v]) for c in outc]
            
            # channels that is remained to be added
            remaining_channels = out_channels[v]-sum(outc)
            
            # do not change
            outc[torch.argmin(torch.tensor(dilations_))]+=remaining_channels
            
            # Store the conv layers for each output channel with different dilations.
            convs_ = []
            for i in range(len(outc)):
                ks = torch.tensor(kernel_size)
                # actual conv kernel size with dilation

                for d in range(dimensions):
                    if ks[d]%2==0:
                        if v==0:
                            assert stride_[d]!=1,f"Impossible to use kernel_size={ks} with stride=1 to get same-shaped tensor"
                        elif stride_[d]==1:
                            ks[d]-=1
                        
                ks_with_dilation = ks + (ks - 1) * (dil[i] - 1)
                compensation = torch.zeros_like(ks_with_dilation)
                for d in range(dimensions):
                    if ks_with_dilation[d]%2==0:
                        compensation[d] = -1
                    else:
                        compensation[d] = 0
                
                conv_kwargs = dict(
                    in_channels=in_ch,
                    out_channels=outc[i],
                    kernel_size=ks.tolist(),
                    padding = (ks_with_dilation // 2 + compensation).tolist(),
                    dilation=dil[i],
                    stride=stride_.tolist(),
                    padding_mode=padding_mode,
                    device=device
                )

                conv__ = x_corr_conv_impl
                # for downsampling use convolutions to extract features
                if self._is_transpose_conv:
                    conv_kwargs['output_padding'] = torch.zeros_like(stride_)
                    for d in range(dimensions):
                        if stride_[d]!=1:
                            conv_kwargs['output_padding'][d]=stride_[d] - 1 + compensation[d]
                    conv_kwargs['output_padding'] = conv_kwargs['output_padding'].tolist()
                    conv__ = x_corr_conv_impl_T
                convs_.append(conv__(**conv_kwargs))
                
            conv = torch.nn.ModuleList(convs_)
            self.convs.append(conv)
            
            #optionally add normalization
            self.norms.append(norm_impl(out_channels[v]).to(device))
            
        self.convs = torch.nn.ModuleList(self.convs)
        self.norms = torch.nn.ModuleList(self.norms)
        
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
        self.disable_residual=disable_residual
        if disable_residual:
            self.x_linear = ConstModule()
            self.alpha = torch.tensor(1.0,device=device)
        else:
            self.alpha = torch.nn.Parameter(torch.tensor(0.0,device=device))
    
    @torch.jit.ignore
    def _conv_x_linear(self, in_channels, out_channels, stride, norm_impl, x_corr_conv_impl,x_corr_conv_impl_T,device):
        # compute x_size correction convolution arguments so we could do residual addition when we have changed
        # number of channels or some stride
        # correct_x_ksize = [1]*self.dimensions
        correct_x_ksize = self.kernel_size
        
        correct_x_ksize = torch.tensor(correct_x_ksize)
        correct_x_padding= correct_x_ksize // 2
        compensation = torch.zeros_like(correct_x_ksize)
        for d in range(self.dimensions):
            if correct_x_ksize[d]%2==0:
                compensation[d] = -1
            else:
                compensation[d] = 0
                
        correct_x_dilation = 1
        stride = torch.tensor(stride)
        # make cheap downscale
        x_corr_kwargs=dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = correct_x_ksize.tolist(),
            dilation=correct_x_dilation,
            stride = stride.tolist(),
            padding = (correct_x_padding+compensation).tolist(),
            padding_mode=self.padding_mode,
            device=device,
            # groups=math.gcd(in_channels,out_channels)
        )
        
        x_conv_impl = x_corr_conv_impl
        if self._is_transpose_conv:
            x_conv_impl = x_corr_conv_impl_T
            x_corr_kwargs['output_padding'] = (stride - 1 + compensation).tolist()
            x_corr_kwargs['groups'] = 1
        
        # if we have different output tensor size, apply linear x_linearion
        # to make sure we can add it with output
        if any([s>1 for s in stride]) or in_channels!=out_channels:
            # there is many ways to linearly downsample x, but max pool with conv2d works best of all
            self.x_linear = \
                torch.nn.Sequential(
                    x_conv_impl(**x_corr_kwargs),
                    norm_impl(out_channels).to(device)
                )
        else:
            self.x_linear = torch.nn.Identity()
        
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
        out = x
        for convs,norm,act in zip(self.convs,self.norms,self.activation):
            results = [conv(out) for conv in convs]
            out = torch.cat(results, dim=1)
            out = act(norm(out))
            out = self.dropout(out)
        
        out_linear = self.x_linear(x)
        # out_linear = resize_tensor(x,out.shape[1:])
        return self.alpha*(out)+out_linear
    
    # to make current block work as transpose (which will upscale input tensor) just use different conv2d implementation
    def transpose(self):
        """
        Creates a transposed (upsampling) version of the current ResidualBlock.

        This method returns a new `ResidualBlock` instance configured with
        transposed convolution implementation, enabling the block to perform upsampling operations.

        Returns:
            ResidualBlock: A new `ResidualBlock` instance configured for transposed convolutions.
        """
        return ResidualBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size = self.kernel_size,
            stride = self.stride,
            dilation = self.dilation,
            dropout=self._dropout_p,
            activation = self._activation_func,
            normalization=self.normalization,
            is_transpose=True,
            dimensions=self.dimensions,
            padding_mode="zeros",
            device=self.__init_device,
            disable_residual=self.disable_residual
        )