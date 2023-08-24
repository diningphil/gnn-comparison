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
# default pytorch version is 1.4.0
PYTORCH_VERSION=2.0.1

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.4.0 are cpu, cu92, cu100, cu101
CUDA_VERSION=11.7.0

# create virtual environment and activate it
conda create --name gnn-comparison python=3.10.9 -y
conda activate gnn-comparison

# install pytorch
conda install -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit
conda install -c pytorch pytorch
