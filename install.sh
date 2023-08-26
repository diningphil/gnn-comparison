#
# Copyright (C) 2020 University of Pisa
# Copyright (C) 2023 University of Pisa, University of Vienna
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
# Default Python version
PYTHON_VERSION=python3.10

# default pytorch version is 2.0.1
# available options are 1.4.0 and 2.0.1
PYTORCH_VERSION=${1:-"2.0.1"}

# Default Torch Geometric version
TORCH_GEOMETRIC_VERSION=2.3.1

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for pytorch 2.0.1 are cpu, and cu117
# available options for pytorch 1.4.0 are cpu, cu92, cu100, and cu101
CUDA_VERSION=${2:-"cpu"}

echo "Using Python version: ${PYTHON_VERSION}"
echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION} support"
echo "Torch Geometric version: ${TORCH_GEOMETRIC_VERSION}"

# create virtual environment and activate it
$PYTHON_VERSION -m pip install --user virtualenv
$PYTHON_VERSION -m venv ~/.venv/gnn-comparison
source ~/.venv/gnn-comparison/bin/activate

pip install build wheel

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cpu
elif [[ "$CUDA_VERSION" == 'cu116' ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu116
elif [[ "$CUDA_VERSION" == 'cu117' ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu117
elif [[ "$CUDA_VERSION" == 'cu118' ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu118
fi

# install torch-geometric
pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}

pip install PyYAML==6.0.1