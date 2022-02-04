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
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def get_max_num_nodes(dataset_str):
    import datasets
    dataset = getattr(datasets, dataset_str)()

    max_num_nodes = -1
    for d in dataset.dataset:
        max_num_nodes = max(max_num_nodes, d.num_nodes)
    return max_num_nodes
