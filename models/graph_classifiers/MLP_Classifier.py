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
import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLPClassifier(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MLPClassifier, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_global = Linear(dim_features, hidden_units)
        self.out = Linear(hidden_units, dim_target)

    def forward(self, x, batch):
        return self.out(F.relu(self.fc_global(x)))
