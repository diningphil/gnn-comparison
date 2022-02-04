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
from torch_geometric.nn import global_add_pool


class DeepMultisets(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(DeepMultisets, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_vertex = Linear(dim_features, hidden_units)
        self.fc_global1 = Linear(hidden_units, hidden_units)
        self.fc_global2 = Linear(hidden_units, dim_target)

    def forward(self, data):
        x, batch = data.x, data.batch

        x = F.relu(self.fc_vertex(x))
        x = global_add_pool(x, batch)  # sums all vertex embeddings belonging to the same graph!
        x = F.relu(self.fc_global1(x))
        x = self.fc_global2(x)
        return x

