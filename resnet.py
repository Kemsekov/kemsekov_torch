from residual import *

class ResidualNetwork(torch.nn.Module):
    def __init__(self,in_channels=3, classes = 100, block_sizes=[2,2,2,2,2], final_pool_size = 1):
        super().__init__()
        conv_impl = BSConvU
        self.block1 = ResidualBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size= 3,
            stride = 2,
            dilation=1,
            repeats=block_sizes[0],
            conv2d_impl=conv_impl
        )
        self.block1=torch.nn.Sequential(self.block1,SEModule(64))
        
        self.block2 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride = 2,
            dilation= 1,
            repeats=block_sizes[1],
            conv2d_impl=conv_impl
        )
        self.block2=torch.nn.Sequential(self.block2,SEModule(128))
        
        self.block3 = ResidualBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride = 2,
            dilation=[1]*64+[2]*32+[3]*32,
            repeats=block_sizes[2],
            conv2d_impl=conv_impl
        )
        self.block3=torch.nn.Sequential(self.block3,SEModule(128))

        
        self.block4 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride = 2,
            dilation=[1]*128+[2]*64+[3]*64,
            repeats=block_sizes[3],
            conv2d_impl=conv_impl
        )
        self.block4=torch.nn.Sequential(self.block4,SEModule(256))
        
        
        self.block5 = ResidualBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride = 2,
            dilation=[1]*256+[2]*128+[3]*128,
            repeats=block_sizes[4],
            conv2d_impl=conv_impl
        )
        self.block5=torch.nn.Sequential(self.block5,SEModule(512))
        
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((final_pool_size,final_pool_size))
        self.fc = torch.nn.Linear(512*final_pool_size**2,classes)
        
    def forward(self,x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = self.block5(x)
        # print(x.shape)
        x = self.avg_pool(x)
        # print(x.shape)
        x = self.fc(x.flatten(1))
        
        return x
