import copy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.ema.to(self.device)
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and isinstance(v, torch.Tensor):
                v.copy_(msd[k] * (1.0 - self.decay) + v * self.decay)
