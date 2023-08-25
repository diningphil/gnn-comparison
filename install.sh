#
# Copyright (C)  2020  University of Pisa
# Copyright (C)  2023  University of Vienna
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
# Find the path to the Conda binary
CONDA_BIN_PATH=$(which conda)

# Extract the path to the Conda installation
CONDA_DIR_PATH=$(dirname $(dirname $CONDA_BIN_PATH))

# Source the Conda initialization script
source $CONDA_DIR_PATH/etc/profile.d/conda.sh

# default pytorch version is 2.0.1
# available options are 1.4.0 and 2.0.1
PYTORCH_VERSION=${1:-"2.0.1"}

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for pytorch 2.0.1 are cpu, and cu117
# available options for pytorch 1.4.0 are cpu, cu92, cu100, and cu101
CUDA_VERSION=${2:-"cpu"}


# create virtual environment and activate it
conda create --name gnn-comparison python=3.10 -y
conda activate gnn-comparison

# install requirements
pip install -r requirements.txt

# install pytorch
case "$CUDA_VERSION" in
  "cpu")
    conda install pytorch==${PYTORCH_VERSION} cpuonly -c pytorch -y
    ;;
  "cu92")
    conda install pytorch==${PYTORCH_VERSION} cudatoolkit=9.2 -c pytorch -y
    ;;
  "cu100")
    conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.0 -c pytorch -y
    ;;
  "cu101")
    conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.1 -c pytorch -y
    ;;
  "cu117")
    conda install -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit
    conda install pytorch==${PYTORCH_VERSION} cudatoolkit=11.7.0 -c pytorch -y
    ;;
  *)
    echo "Error: Unsupported CUDA version: ${CUDA_VERSION}. Exiting."
    exit 1
    ;;
esac

# install torch-geometric
pip install torch-geometric
