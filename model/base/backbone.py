import torch.nn as nn
from torch import chunk, cat

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
        self.conv = ConvModule(1, 1, 0, c, c)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        return cat([x1, x2, x3, x4], dim = 1)

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

class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):

        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        self.conv1 = ConvModule(3, 2, 1, 3, 64 * w)
        self.conv2 = ConvModule(3, 2, 1, 64 * w, 128 * w)
        self.conv3 = ConvModule(3, 2, 1, 128 * w, 256 * w)
        self.conv4 = ConvModule(3, 2, 1, 256 * w, 512 * w)
        self.conv5 = ConvModule(3, 2, 1, 512 * w, 512 * w * r)

        self.csplayer1 = CSPLayer_2Conv(n, 128 * w, 128 * w, True)
        self.csplayer2 = CSPLayer_2Conv(2 * n, 256 * w, 256 * w, True)
        self.csplayer3 = CSPLayer_2Conv(2 * n, 512 * w, 512 * w, True)
        self.csplayer4 = CSPLayer_2Conv(n, 512 * w * r, 512 * w * r, True)

        self.sppf = SPPF(512 * w * r)

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

        import pdb; pdb.set_trace()

        # Stem Layer
        stemLayer = self.conv1(x)
        
        # Stage Layer 1
        feat0 = self.conv2(stemLayer)
        feat0 = self.csplayer1(feat0)

        # Stage Layer 2
        feat1 = self.conv3(feat0)
        feat1 = self.csplayer2(feat1)

        # Stage Layer 3
        feat2 = self.conv4(feat1)
        feat2 = self.csplayer3(feat2)

        # Stage Layer 4
        feat3 = self.conv5(feat2)
        feat3 = self.csplayer4(feat3)
        feat3 = self.sppf(feat3)

        return 

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