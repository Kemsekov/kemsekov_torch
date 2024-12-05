from typing import List
from residual import *

class Encoder(torch.nn.Module):
    def __init__(self, in_channels_, out_channels_, dilations, block_sizes, downs_conv2d_impl):
        super().__init__()
        downs_list = []
        for i in range(len(block_sizes)):
            down_i = ResidualBlock(
                in_channels=in_channels_[i],
                out_channels=out_channels_[i],
                kernel_size= 3,
                stride = 2,
                dilation=dilations[i],
                repeats=block_sizes[i],
                conv2d_impl=downs_conv2d_impl[i]
            )
            down_i = torch.nn.Sequential(down_i,SEModule(out_channels_[i]))
            downs_list.append(down_i)
        self.downs =        torch.nn.ModuleList(downs_list[:-1])
        self.down5 = downs_list[-1]

    def forward(self, x):
        skip_connections = []
        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        x = self.down5(x)
        return x, skip_connections
    def forward_without_skip(self,x):
        """makes same forward but do not keep track of skip connections"""
        for down in self.downs:
            x = down(x)
        x = self.down5(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, up_in_channels, up_out_channels, up_block_sizes, ups_conv2d_impl, output_scale, out_channels):
        super().__init__()
        ups = []
        conv1x1s = []
        for i in range(len(ups_conv2d_impl)):
            up_i = ResidualBlock(
                in_channels=up_in_channels[i],
                out_channels=up_out_channels[i],
                kernel_size=3,
                stride = 2,
                dilation=1,
                repeats=up_block_sizes[i],
                conv2d_impl=ups_conv2d_impl[i]
            )
            up_i = torch.nn.Sequential(up_i,SEModule(up_out_channels[i]))
            conv1x1_i = torch.nn.Conv2d(up_out_channels[i]*2, up_out_channels[i], kernel_size=1)

            ups.append(up_i)
            conv1x1s.append(conv1x1_i)

        self.ups =          torch.nn.ModuleList(ups)
        self.up_1x1_convs = torch.nn.ModuleList(conv1x1s)

        if output_scale<1:
            kernel_size = 3
            # for smaller scales we need to use larger kernel
            if output_scale<0.25:
                kernel_size = 5
            self.up5 = ResidualBlock(
                in_channels=up_in_channels[4],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride = int(0.5/output_scale),
                dilation=1,
                repeats=up_block_sizes[4],
                conv2d_impl=nn.Conv2d
            )
        
        if output_scale==1:
            self.up5 = ResidualBlock(
                in_channels=up_in_channels[4],
                out_channels=out_channels,
                kernel_size=3,
                stride = 2,
                dilation=1,
                repeats=up_block_sizes[4],
                conv2d_impl=torch.nn.ConvTranspose2d
            )

    def forward(self,x: torch.Tensor,skip_connections : List[torch.Tensor]):
        # Upsampling path
        for i, (up,conv_1x1) in enumerate(zip(self.ups,self.up_1x1_convs)):
            x = up(x)
            if len(skip_connections)!=0:
                # Concatenate the corresponding skip connection (from the downsampling path)
                skip = skip_connections[-(i + 1)]
                
                # here skip needs to be reshaped to x size before making concat
                x = torch.cat((x, skip), dim=1)
                
                # to decrease num of channels
                x = conv_1x1(x)
        x = self.up5(x)
        return x
    
class ResidualUnet(torch.nn.Module):
    # output_scale must be power of 2: 0.125 0.25 0.5 1 2 4 etc
    # it defines what size of output tensor should be relative to input tensor
    def __init__(self,in_channels=3, out_channels = 1, block_sizes=[2,2,2,2,2],output_scale = 1):
        super().__init__()
        assert output_scale>=0.125 and output_scale<=1, "output_scale must be in range [0.125,1]"

        in_channels_ = [in_channels,64,128,128,256]
        out_channels_ = [64,128,128,256,256]
        dilations=[
            1,
            1,
            [1]*64+[2]*64,
            [1]*128+[2]*64+[3]*64,
            [1]*128+[2]*64+[3]*64
        ]

        downs_conv2d_impl = [
            [nn.Conv2d]+[BSConvU]*(block_sizes[0]-1),
            [nn.Conv2d]+[BSConvU]*(block_sizes[1]-1),
            [nn.Conv2d]+[BSConvU]*(block_sizes[2]-1),
            [nn.Conv2d]+[BSConvU]*(block_sizes[3]-1),
            [nn.Conv2d]+[BSConvU]*(block_sizes[4]-1),
        ]

        up_block_sizes = block_sizes[::-1]
        ups_conv2d_impl = [
            [nn.ConvTranspose2d]+[BSConvU]*(up_block_sizes[0]-1),
            [nn.ConvTranspose2d]+[BSConvU]*(up_block_sizes[1]-1),
            [nn.ConvTranspose2d]+[BSConvU]*(up_block_sizes[2]-1),
            [nn.ConvTranspose2d]+[BSConvU]*(up_block_sizes[3]-1),
        ]
        up_in_channels = out_channels_[::-1]
        up_out_channels = in_channels_ [::-1]
        self.encoder = Encoder(in_channels_,out_channels_,dilations,block_sizes,downs_conv2d_impl)
        self.decoder = Decoder(up_in_channels,up_out_channels,up_block_sizes,ups_conv2d_impl,output_scale,out_channels)

    def forward(self, x):
        x,skip = self.encoder(x)
        x = self.decoder(x,skip)
        return x
