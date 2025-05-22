import torch.nn as nn
from .components import Conv, C2f, SPPF


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):

        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        self.l0 = Conv(3, int(64 * w), 3, 2, 1)
        self.l1 = Conv(int(64 * w), int(128 * w), 3, 2, 1)
        self.l3 = Conv(int(128 * w), int(256 * w), 3, 2, 1)
        self.l5 = Conv(int(256 * w), int(512 * w), 3, 2, 1)
        self.l7 = Conv(int(512 * w), int(512 * w * r), 3, 2, 1)

        self.l2 = C2f(int(128 * w), int(128 * w), n, True)
        self.l4 = C2f(int(256 * w), int(256 * w), 2 * n, True)
        self.l6 = C2f(int(512 * w), int(512 * w), 2 * n, True)
        self.l8 = C2f(int(512 * w * r), int(512 * w * r), n, True)

        self.l9 = SPPF(int(512 * w * r), int(512 * w * r))

        # raise NotImplementedError("Backbone::__init__")

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """

        # Stem Layer
        stemLayer = self.l0(x)
        
        # Stage Layer 1
        feat0 = self.l1(stemLayer)
        feat0 = self.l2(feat0)

        # Stage Layer 2
        feat1 = self.l3(feat0)
        feat1 = self.l4(feat1)

        # Stage Layer 3
        feat2 = self.l5(feat1)
        feat2 = self.l6(feat2)

        # Stage Layer 4
        feat3 = self.l7(feat2)
        feat3 = self.l8(feat3)
        feat3 = self.l9(feat3)

        return feat0, feat1, feat2, feat3

        # raise NotImplementedError("Backbone::forward")

"""
def runConvModule(x, k, s, p, c):  # 运行ConvModule，参数直接按“数据+k、s、p、c(out)”的顺序输入
    return  nn.Sequential(
                nn.Conv2d(x.shape[1], c, k, stride=s, padding=p),
                nn.BatchNorm2d(c),
                nn.SiLU()
                )(x)

def runSPPF(x):
    maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    x1 = runConvModule(x, 1, 1, 0, x.shape[1])
    x2 = maxpool(x1)
    x3 = maxpool(x2)
    x4 = maxpool(x3)
    return cat([x1, x2, x3, x4], dim = 1)

def runDarknetBottleneck(x, add):
    out = runConvModule(x, 3, 1, 1, 0.5 * x.shape[1])
    out = runConvModule(out, 3, 1, 1, x.shape[1])
    if add == True:
        out += x
    return out  

def runCSPLayer_2Conv(x, add, n, c):    # 运行CSP层
    x = runConvModule(x, 1, 1, 0, c)
    out, res = chunk(x, 2, dim=1)
    out = cat([out, res], dim = 1)
    for _ in range(n):
        res = runDarknetBottleneck(res, add)
        out = cat([out, res], dim = 1)
    out = runConvModule(out, 1, 1, 0, c)
    return out
"""

"""
        self.l0 = ConvModule(3, 2, 1, 3, 64 * w)
        self.l1 = ConvModule(3, 2, 1, 64 * w, 128 * w)
        self.l3 = ConvModule(3, 2, 1, 128 * w, 256 * w)
        self.l5 = ConvModule(3, 2, 1, 256 * w, 512 * w)
        self.l7 = ConvModule(3, 2, 1, 512 * w, 512 * w * r)

        self.l2 = CSPLayer_2Conv(n, 128 * w, 128 * w, True)
        self.l4 = CSPLayer_2Conv(2 * n, 256 * w, 256 * w, True)
        self.l6 = CSPLayer_2Conv(2 * n, 512 * w, 512 * w, True)
        self.l8 = CSPLayer_2Conv(n, 512 * w * r, 512 * w * r, True)

        self.l9 = SPPF(512 * w * r)
"""

"""
class ConvModule(nn.Module):
    def __init__(self, k, s, p, c_in, c_out):
        super().__init__()

        c_in = int(c_in)
        c_out = int(c_out)
        
        self.conv = nn.Sequential(
                        nn.Conv2d(c_in, c_out, k, stride=s, padding=p),
                        nn.BatchNorm2d(c_out),
                        nn.SiLU()
                        )

    def forward(self, x):
        return self.conv(x)

class SPPF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvModule(1, 1, 0, c, c)
        self.conv2 = ConvModule(1, 1, 0, 4 * c, c)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        out = cat([x1, x2, x3, x4], dim = 1)
        out = self.conv2(out)
        
        return out

class DarknetBottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvModule(3, 1, 1, c, 0.5 * c)
        self.conv2 = ConvModule(3, 1, 1, 0.5 * c, c)

    def forward(self, x, add):
        out = self.conv1(x)
        out = self.conv2(out)
        if add == True:
            out += x
        return out  

class CSPLayer_2Conv(nn.Module):
    def __init__(self, n, c_in, c_out, add):
        super().__init__()
        self.n = n
        self.add = add

        self.conv1 = ConvModule(1, 1, 0, c_in, c_out)
        self.conv2 = ConvModule(1, 1, 0, 0.5 * c_out * (n+2), c_out)
        self.darknetbottleneck = DarknetBottleneck(0.5 * c_out)

    def forward(self, x):
        x = self.conv1(x)
        out, res = chunk(x, 2, dim=1)
        out = cat([out, res], dim = 1)
        for _ in range(self.n):
            res = self.darknetbottleneck(res, self.add)
            out = cat([out, res], dim = 1)
        out = self.conv2(out)
        return out

"""