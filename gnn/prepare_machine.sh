#!/bin/bash

set -e
set -x

# Install pip
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip

# Install torch and prerequisites
pip3 install pyyaml gitpython pandas networkx tqdm matplotlib nni==2.10.1 --break-system-packages
pip3 install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --break-system-packages
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html --break-system-packages
pip3 install torch_geometric --break-system-packages
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
nvcc --version
