# default pytorch version is 1.4.0
PYTORCH_VERSION=1.4.0

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.4.0 are cpu, cu92, cu100, cu101
CUDA_VERSION=${1:-cpu}

# create virtual environment and activate it
conda create --name gnn-comparison python=3.7 -y
conda activate gnn-comparison

# install requirements
pip install -r requirements.txt

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then 
  conda install pytorch==${PYTORCH_VERSION} cpuonly -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu92' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=9.2 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu100' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.0 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu101' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.1 -c pytorch -y
fi

# install torch-geometric dependencies
pip install torch-scatter==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-sparse==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-cluster==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-spline-conv==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-geometric==1.4.2
