from torchvision.ops import DeformConv2d
import torch.nn as nn
import torch

# drop-in replacement for ordinary conv
class Conv2dDeform(nn.Module):
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
        mask = torch.sigmoid(self.mask_conv(x))  # Ensure mask values are between 0 and 1
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset, mask)
        return self.x_res(x)+out
