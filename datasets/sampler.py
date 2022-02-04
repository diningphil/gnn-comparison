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
from torch.utils.data import sampler


class RandomSampler(sampler.RandomSampler):
    """
    This sampler saves the random permutation applied to the training data,
    so it is available for further use (e.g. for saving).
    The permutation is saved in the 'permutation' attribute.
    The DataLoader can now be instantiated as follows:

    >>> data = Dataset()
    >>> dataloader = DataLoader(dataset=data, batch_size=32, shuffle=False, sampler=RandomSampler(data))
    >>> for batch in dataloader:
    >>>     print(batch)
    >>> print(dataloader.sampler.permutation)

    For convenience, one can create a method in the dataloader class to access the random permutation directly, e.g:

    class MyDataLoader(DataLoader):
        ...
        def get_permutation(self):
            return self.sampler.permutation
        ...
    """

    def __init__(self, data_source, num_samples=None, replacement=False):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples)
        self.permutation = None

    def __iter__(self):
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)
