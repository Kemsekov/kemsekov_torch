import torch.nn.functional as F

# change tensor shape
class Interpolate(nn.Module):
    def __init__(self, scale_factor=None, size=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode, align_corners=self.align_corners)
