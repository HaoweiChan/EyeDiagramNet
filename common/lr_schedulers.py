import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmRestartsWithWarmupDecay(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        warmup_epochs=0,
        decay_factor=1.0
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.decay_factor = decay_factor
        self.T_cur = 0
        self.T_i = T_0
        self.max_lrs = [group['lr'] for group in optimizer.param_groups]
        super(CosineAnnealingWarmRestartsWithWarmupDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.max_lrs
            ]

        # If the current restart period is complete, reset and update T_i and max_lrs
        if self.T_cur == self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            self.max_lrs = [lr * self.decay_factor for lr in self.max_lrs]

        cos_inner = math.pi * (self.T_cur / self.T_i)
        cos_out = math.cos(cos_inner)
        lrs = [
            self.eta_min + (max_lr - self.eta_min) / 2 * (1 + cos_out)
            for max_lr in self.max_lrs
        ]
        return lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Once we’re past warmup, advance the T_cur counter each step
        if self.last_epoch >= self.warmup_epochs:
            self.T_cur += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
