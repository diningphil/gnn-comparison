#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
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
