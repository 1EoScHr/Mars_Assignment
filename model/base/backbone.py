import torch.nn as nn
from torch import chunk, cat

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
        self.r = r
        self.n = n

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
        stemLayer = runConvModule(x, self.kernelSize, self.stride, 1, 64 * self.w)
        
        # Stage Layer 1
        feat0 = runConvModule(stemLayer, self.kernelSize, self.stride, 1, 128 * self.w)
        feat0 = runCSPLayer_2Conv(feat0, True, self.n, 128 * self.w)

        # Stage Layer 2
        feat1 = runConvModule(feat0, self.kernelSize, self.stride, 1, 256 * self.w)
        feat1 = runCSPLayer_2Conv(feat1, True, 2 * self.n, 256 * self.w)

        # Stage Layer 3
        feat2 = runConvModule(feat1, self.kernelSize, self.stride, 1, 512 * self.w)
        feat2 = runCSPLayer_2Conv(feat2, True, 6 * self.n, 512 * self.w)

        # Stage Layer 4
        feat3 = runConvModule(feat2, self.kernelSize, self.stride, 1, 512 * self.w * self.r)
        feat3 = runCSPLayer_2Conv(feat2, True, 3 * self.n, 512 * self.w * self.r)
        feat3 = runSPPF(feat3)

        return 

        # raise NotImplementedError("Backbone::forward")
