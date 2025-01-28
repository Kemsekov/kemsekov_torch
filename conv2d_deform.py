from torchvision.ops import DeformConv2d
import torch.nn as nn
import torch

class Conv2dDeform(nn.Module):
    """
    A drop-in replacement for a standard 2D convolutional layer that incorporates deformable convolution.

    This module enhances the flexibility of the receptive fields by allowing spatial deformation, which can improve
    the modeling of geometric transformations in convolutional neural networks.

    Attributes:
        offset_conv (nn.Conv2d): Convolutional layer that predicts the offsets for the deformable convolution.
        mask_conv (nn.Conv2d): Convolutional layer that predicts the modulation mask for the deformable convolution.
        deform_conv (DeformConv2d): The deformable convolutional layer.
        x_res (nn.Conv2d): Residual convolutional layer to match the input dimensions to the output dimensions.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple, optional): Size of the convolving kernel. Default is 3.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default is 1.
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default is 'zeros'.
        dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,padding_mode='zeros',dilation=1):
        super(Conv2dDeform, self).__init__()
        offset_out_ch = 2 * kernel_size * kernel_size
        mask_out_ch = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, offset_out_ch, kernel_size=kernel_size, stride=stride, padding=padding,padding_mode=padding_mode,dilation=dilation)
        self.mask_conv = nn.Conv2d(in_channels, mask_out_ch, kernel_size=kernel_size, stride=stride, padding=padding,padding_mode=padding_mode,dilation=dilation)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation)
        if kernel_size//2 == padding and stride==1:
            self.x_res = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.x_res = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            
    def forward(self, x):
        """
        Forward pass of the Conv2dDeform layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where:
                N = batch size
                C = number of channels
                H = height of the input
                W = width of the input

        Returns:
            torch.Tensor: Output tensor after applying deformable convolution and residual connection.
        """
        mask = torch.sigmoid(self.mask_conv(x))  # Ensure mask values are between 0 and 1
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset, mask)
        return self.x_res(x)+out
