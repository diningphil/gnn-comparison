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


class DatasetGetter:

    def __init__(self, outer_k=None, inner_k=None):
        self.outer_k = outer_k
        self.inner_k = inner_k

    def set_inner_k(self, k):
        self.inner_k = k

    def get_train_val(self, dataset, batch_size, shuffle=True):
        return dataset.get_model_selection_fold(self.outer_k, self.inner_k, batch_size, shuffle)

    def get_test(self, dataset, batch_size, shuffle=True):
        return dataset.get_test_fold(self.outer_k, batch_size, shuffle)