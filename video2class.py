from kemsekov_torch.residual import ResidualBlock
from kemsekov_torch.conv_modules import SCSEModule
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self, dim1 = -1,dim2 = -2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self,x):
        return x.transpose(self.dim1,self.dim2)

class Video2Class(nn.Sequential):
    def __init__(self,in_channels = 3,num_classes = 3,dropout_p=0.1,block_size=2):
        """
        accepts tensor:
            batch, images_count,image_channels, width, height

        outputs logits per each image:
            batch,images_count,num_classes
        
        This network spatially convolve video frames using 3d convolutions
        """
        super().__init__()
        block_size = 2
        model = [
            Transpose(1,2),
            nn.BatchNorm3d(in_channels),
            ResidualBlock(
                in_channels=in_channels,
                out_channels=[64]*block_size,
                dimensions=3
            ),
            nn.MaxPool3d((1,2,2)),
            SCSEModule(64,dimensions=3),
            nn.Dropout3d(dropout_p),
            ResidualBlock(
                in_channels=64,
                out_channels=[128]*block_size,
                dilation=[1]*64+[2]*64,
                dimensions=3
            ),
            nn.MaxPool3d((1,2,2)),
            SCSEModule(128,dimensions=3),
            nn.Dropout3d(dropout_p),
            ResidualBlock(
                in_channels=128,
                out_channels=[256]*block_size,
                dilation=[1]*128+[2]*128,
                dimensions=3
            ),
            nn.MaxPool3d((1,2,2)),
            SCSEModule(256,dimensions=3),
            nn.Dropout3d(dropout_p),
            ResidualBlock(
                in_channels=256,
                out_channels=[512]*block_size,
                dilation=[1]*256+[2]*256,
                dimensions=3
            ),
            nn.MaxPool3d((1,2,2)),
            SCSEModule(512,dimensions=3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(2),
            Transpose(),
            nn.Linear(512,num_classes)
        ]
        self.extend(model)