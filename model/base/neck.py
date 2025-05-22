import torch
import torch.nn as nn
from .components import Conv, C2f


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

        self.l12 = C2f(int(512 * w * (1+r)), int(512 * w), n, False)
        self.l15 = C2f(int(768 * w), int(256 * w), n, False)
        self.l18 = C2f(int(768 * w), int(512 * w), n, False)
        self.l21 = C2f(int(512 * w * (1+r)), int(512 * w * r), n, False)

        self.l16 = Conv(int(256 * w), int(256 * w), 3, 2, 1)
        self.l19 = Conv(int(512 * w), int(512 * w), 3, 2, 1)

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
        topDownlayer_2 = self.l12(topDownlayer_2)

        topDownlayer_1 = torch.cat([runUpsample(topDownlayer_2), feat1], dim = 1)
        topDownlayer_1 = self.l15(topDownlayer_1)

        bottomUplayer_0 = self.l16(topDownlayer_1)
        bottomUplayer_0 = torch.cat([bottomUplayer_0, topDownlayer_2], dim = 1)
        bottomUplayer_0 = self.l18(bottomUplayer_0)

        bottomUplayer_1 = self.l19(bottomUplayer_0)
        bottomUplayer_1 = torch.cat([bottomUplayer_1, feat3], dim = 1)
        bottomUplayer_1 = self.l21(bottomUplayer_1)

        return topDownlayer_2, topDownlayer_1, bottomUplayer_0, bottomUplayer_1

        # raise NotImplementedError("Neck::forward")

"""
        self.l12 = CSPLayer_2Conv(n, 512 * w * (1+r), 512 * w, False) # t means top
        self.l15 = CSPLayer_2Conv(n, 768 * w, 256 * w, False)
        self.l18 = CSPLayer_2Conv(n, 768 * w, 512 * w ,False)         # b means bottom
        self.l21 = CSPLayer_2Conv(n, 512 * w * (1+r), 512 * w * r, False)

        self.l16 = ConvModule(3, 2, 1, 256 * w, 256 * w)
        self.l19 = ConvModule(3, 2, 1, 512 * w, 512 * w)
"""