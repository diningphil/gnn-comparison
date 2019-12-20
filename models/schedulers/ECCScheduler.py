from torch.optim.lr_scheduler import StepLR


class ECCLR(StepLR):

    def __init__(self, optimizer, step_size=1, gamma=0.1, last_epoch=-1):
        self.step_size = step_size  # does not matter
        self.gamma = gamma
        super(ECCLR, self).__init__(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch in [25, 35, 45]:
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
