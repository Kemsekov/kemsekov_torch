import inspect
import math
from typing import Literal
import torch
import torch.nn as nn
from conv_modules import *
from common_modules import *

class Residual(torch.nn.Module):
    """
    Residual module that sums outputs of module with it's input. It supports any models that outputs any shape.
    """
    def __init__(self,m):
        """
        Residual module that wraps around module `m`.
        
        This module uses Re-Zero approach to add module output(multiplied by `alpha`) with it's inputs.
        
        When module output shape != input tensor shape, it uses nearest-exact resize approach to match input shape to output, 
        and performs addition as described.
        
        m - module that takes some input and spits output
        """
        super().__init__()
        self.m = m
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        out = self.m(x)
        x_resize = resize_tensor(x,out.shape[1:])
        return self.alpha*out+x_resize

class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        dilation = 1,
        activation=torch.nn.ReLU,
        normalization : Literal['batch','instance','group',None] = 'batch',
        dimensions : Literal[1,2,3] = 2,
        is_transpose = False,
        padding_mode : Literal['constant', 'reflect', 'replicate', 'circular']="replicate"
    ):
        super().__init__()
        
        x_corr_conv_impl = [nn.Conv1d,nn.Conv2d,nn.Conv3d][dimensions-1]
        x_corr_conv_impl_T = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][dimensions-1]
            
        if not isinstance(out_channels,list):
            out_channels=[out_channels]
                
        repeats=len(out_channels)
        self.padding_mode=padding_mode
        self._is_transpose_conv = is_transpose

        self.normalization = normalization
        if not isinstance(kernel_size,list):
            kernel_size=[kernel_size]

        self.kernel_size = kernel_size
        
        
        norm_impl = get_normalization_from_name(dimensions,normalization)
        self.dimensions=dimensions
        
        if not isinstance(dilation,list):
            dilation=[dilation]*out_channels[0]

        if len(kernel_size)==1 and len(dilation)!=1:
            kernel_size=kernel_size*len(dilation)
        
        if len(kernel_size)!=1 and len(dilation)==1:
            dilation=dilation*len(kernel_size)


        # assert len(dilation) == out_channels[0], f"Number of dilations must match the number of output channels at first dim. {len(dilation)} != {out_channels[0]}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self._activation_func = activation
        self.repeats = repeats
        
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
        self.input_resize=nn.ModuleList([torch.nn.Identity()]*repeats)
        
        assert stride%2==0 or stride==1, f"stride must be even or 1. given stride={stride}"
        strides = [1]*repeats
        
        stride_ind=0
        while math.prod(strides)<stride:
            strides[stride_ind%repeats]*=2
            stride_ind+=1
        
        for v in range(repeats):
            # on first repeat block make sure to cast input tensor to output shape
            # and on further repeats just make same-shaped transformations
            in_ch = in_channels if v==0 else out_channels[v-1]
            stride_ = strides[v]

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
                    elif stride_==1:
                        ks-=1
                        
                ks_with_dilation = ks + (ks - 1) * (dil[i] - 1)
                
                if ks_with_dilation%2==0:
                    compensation = -1
                else:
                    compensation=0
                
                conv_kwargs = dict(
                    in_channels=in_ch,
                    out_channels=outc[i],
                    kernel_size=ks,
                    padding = ks_with_dilation // 2 + compensation,
                    dilation=dil[i],
                    stride=stride_,
                    padding_mode=padding_mode
                )

                conv__ = x_corr_conv_impl
                # for downsampling use convolutions to extract features
                if stride_!=1 and self._is_transpose_conv:
                    conv_kwargs['output_padding']=stride_ - 1 + compensation
                    conv__ = x_corr_conv_impl_T
                convs_.append(conv__(**conv_kwargs))
                
            conv = torch.nn.ModuleList(convs_)
            self.convs.append(conv)
            
            #optionally add normalization
            self.norms.append(norm_impl(out_channels[v]))
            
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
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))

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
        for convs,norm,act,resize in zip(self.convs,self.norms,self.activation,self.input_resize):
            out=resize(out)
            results = [conv(out) for conv in convs]
            out = torch.cat(results, dim=1)
            out = act(norm(out))
        x_resize = resize_tensor(x,out.shape[1:])
        return self.alpha*out+x_resize
    
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
            activation = self._activation_func,
            normalization=self.normalization,
            is_transpose=True,
            dimensions=self.dimensions,
            padding_mode="zeros"
        )