import torch
import torch.nn as nn

from .backbone import runConvModule, runCSPLayer_2Conv

def runUpsample(x):
    return nn.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        self.w = w
        self.r = r
        self.n = n

        # raise NotImplementedError("Neck::__init__")

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """

        topDownlayer_2 = torch.cat([runUpsample(feat3), feat2], dim = 1)
        topDownlayer_2 = runCSPLayer_2Conv(topDownlayer_2, False, self.n, 512 * self.w)

        topDownlayer_1 = torch.cat([runUpsample(topDownlayer_2), feat1], dim = 1)
        topDownlayer_1 = runCSPLayer_2Conv(topDownlayer_1, False, self.n, 256 * self.w)

        bottomUplayer_0 = runConvModule(topDownlayer_1, self.kernelSize, self.stride, 1, 256 * self.w)
        bottomUplayer_0 = torch.cat([bottomUplayer_0, topDownlayer_2], dim = 1)
        bottomUplayer_0 = runCSPLayer_2Conv(bottomUplayer_0, False, self.n, 512 * self.w)

        bottomUplayer_1 = runConvModule(bottomUplayer_0, self.kernelSize, self.stride, 1, 512 * self.w)
        bottomUplayer_1 = torch.cat([bottomUplayer_1, feat3], dim = 1)
        bottomUplayer_1 = runCSPLayer_2Conv(bottomUplayer_1, False, self.n, 512 * self.w * self.r)

        return 

        # raise NotImplementedError("Neck::forward")
