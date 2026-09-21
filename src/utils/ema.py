import copy
import torch


class EMAHelper:
    def __init__(self, mu=0.999, device="cpu"):
        self.mu = mu
        self.shadow = {}
        self.device = device
        self.step = 0  # Track total updates for dynamic warmup

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
        self.step = 0

    def update(self, module):
        self.step += 1
        # Dynamic decay: starts fast, asymptotically approaches self.mu
        decay = min(self.mu, (1.0 + self.step) / (10.0 + self.step))

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - decay
                ) * param.data + decay * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module).to(self.device)
        self.ema(module_copy)
        return module_copy
