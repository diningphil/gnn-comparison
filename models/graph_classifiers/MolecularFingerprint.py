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
from torch.nn import ReLU
from torch_geometric.nn import global_add_pool


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']

        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), ReLU(),
                                       torch.nn.Linear(hidden_dim, dim_target), ReLU())

    def forward(self, data):
        return self.mlp(global_add_pool(data.x, data.batch))
