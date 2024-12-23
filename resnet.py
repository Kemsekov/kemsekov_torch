from residual import *

class ResidualNetwork(torch.nn.Module):
    def __init__(self,in_channels=3, classes = 100, block_sizes=[2,2,2,2,2], final_pool_size = 1,dropout_p=0.5):
        super().__init__()
        conv_impl = BSConvU
        self.block1 = ResidualBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size= 3,
            stride = 2,
            dilation=1,
            conv_impl=[conv_impl]*block_sizes[0],
        )
        self.block1=torch.nn.Sequential(self.block1,SEModule(64),nn.Dropout2d(dropout_p))
        
        self.block2 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride = 2,
            dilation= 1,
            conv_impl=[conv_impl]*block_sizes[1],
        )
        self.block2=torch.nn.Sequential(self.block2,SEModule(128),nn.Dropout2d(dropout_p))
        
        self.block3 = ResidualBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride = 2,
            dilation=[1]*96+[2]*32,
            conv_impl=[conv_impl]*block_sizes[2],
        )
        self.block3=torch.nn.Sequential(self.block3,SEModule(128),nn.Dropout2d(dropout_p))

        self.block4 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride = 2,
            dilation=[1]*128+[2]*64+[3]*64,
            conv_impl=[conv_impl]*block_sizes[3],
        )
        self.block4=torch.nn.Sequential(self.block4,SEModule(256),nn.Dropout2d(dropout_p))
        
        self.block5 = ResidualBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride = 2,
            dilation=[1]*256+[2]*128+[3]*128,
            conv_impl=[conv_impl]*block_sizes[4],
        )
        self.block5=torch.nn.Sequential(self.block5,SEModule(512),nn.Dropout2d(dropout_p))
        
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
