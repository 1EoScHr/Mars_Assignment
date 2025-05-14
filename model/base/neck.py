import torch
import torch.nn as nn

from .backbone import ConvModule, CSPLayer_2Conv

def runUpsample(x):
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        self.csplayer_t2 = CSPLayer_2Conv(n, 512 * w * (1+r), 512 * w, False) # t means top
        self.csplayer_t1 = CSPLayer_2Conv(n, 768 * w, 256 * w, False)
        self.csplayer_b0 = CSPLayer_2Conv(n, 768 * w, 512 * w ,False)         # b means bottom
        self.csplayer_b1 = CSPLayer_2Conv(n, 512 * w * (1+r), 512 * w * r, False)

        self.conv_b0 = ConvModule(3, 2, 1, 256 * w, 256 * w)
        self.conv_b1 = ConvModule(3, 2, 1, 512 * w, 512 * w)

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
        topDownlayer_2 = self.csplayer_t2(topDownlayer_2)

        topDownlayer_1 = torch.cat([runUpsample(topDownlayer_2), feat1], dim = 1)
        topDownlayer_1 = self.csplayer_t1(topDownlayer_1)

        bottomUplayer_0 = self.conv_b0(topDownlayer_1)
        bottomUplayer_0 = torch.cat([bottomUplayer_0, topDownlayer_2], dim = 1)
        bottomUplayer_0 = self.csplayer_b0(bottomUplayer_0)

        bottomUplayer_1 = self.conv_b1(bottomUplayer_0)
        bottomUplayer_1 = torch.cat([bottomUplayer_1, feat3], dim = 1)
        bottomUplayer_1 = self.csplayer_b1(bottomUplayer_1)

        return topDownlayer_2, topDownlayer_1, bottomUplayer_0, bottomUplayer_1

        # raise NotImplementedError("Neck::forward")
