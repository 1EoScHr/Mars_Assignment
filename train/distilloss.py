import torch
from overrides import override # this could be removed since Python 3.12
from .loss import DetectionLoss
from .criterion import CriterionCWD

class ResponseLoss(torch.nn.Module):
    def __init__(self, temperature = 1.0, class_indexes = None, reg_max = 16):
        super(ResponseLoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='sum') # 返回sum得来的总损失
        self.temperature = temperature
        self.class_indexes = class_indexes
        self.reg_max = reg_max
        self.class_offset = 4 * reg_max  # 分类通道在预测输出中的起始位置
    
    def forward(self, pred_S, pred_T):
        if self.class_indexes is not None:  # 切片
            pred_S = [p[:, [self.class_offset + i for i in self.class_indexes], :, :] for p in pred_S]
            pred_T = [p[:, [self.class_offset + i for i in self.class_indexes], :, :] for p in pred_T]
        
        total_loss = 0

        for s_feat, t_feat in zip(pred_S, pred_T):
            n,c,h,w = s_feat.shape

            s = torch.log_softmax(s_feat / self.temperature, dim = 1)
            t = torch.softmax(t_feat.detach() / self.temperature, dim = 1)
            loss = self.criterion(s, t)

            loss /= n * c # 保持相同的归一化
            total_loss += loss * (self.temperature ** 2)

        return total_loss

class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CriterionCWD(temperature=mcfg.temperature).to(mcfg.device)
        self.respLoss = ResponseLoss(temperature=mcfg.temperature, class_indexes=mcfg.teacherClassIndexes, reg_max=self.mcfg.regMax)
        # raise NotImplementedError("DistillationDetectionLoss::__init__")

    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0] & rawPreds[1] shape: (
            (B, regMax * 4 + nc, 80, 80),
            (B, regMax * 4 + nc, 40, 40),
            (B, regMax * 4 + nc, 20, 20),
            (B, 128 * w, 160, 160),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
            (B, 512 * w, 40, 40),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
        )
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()
