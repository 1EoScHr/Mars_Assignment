import torch.nn as nn
from torch import chunk

def runConvModule(x, k, s, p, c):  # 运行ConvModule，参数直接按“数据+k、s、p、c(out)”的顺序输入
    return  nn.Sequential(
                nn.Conv2d(x.shape[1], c, k, stride=s, padding=p),
                nn.BatchNorm2d(c),
                nn.SiLU()
                )(x)

def runDarknetBottleneck(x, add):
    pass

def runCSPLayer_2Conv(x, add, n, c):    # 运行CSP层
    x = runConvModule(x, 1, 1, 0, c)
    x1, x2 = chunk(x, 2, dim=1)
    for _ in range(n):
        x1, x2 = runDarknetBottleneck(x1, False)
    x = x1 + x2
    x = runConvModule(x, 1, 1, 0, c)
    return x    

class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2
        
        self.w = w

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
        
        stemLayer = runConvModule(x, self.kernelSize, self.stride, 1, 64 * self.w)
        feat0 = runConvModule(stemLayer, self.kernelSize, self.stride, 1, 128 * self.w)
        feat0 = runCSPLayer_2Conv(feat0, True, )


        # raise NotImplementedError("Backbone::forward")
