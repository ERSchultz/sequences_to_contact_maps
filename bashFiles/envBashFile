#! /bin/bash

# cd ~/sequences_to_contact_maps
# conda create --name seq2contact_pytorch -y
# source activate seq2contact_pytorch
# conda install -y pytorch=1.8.1 torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy gcc-5 -c psi4 -c pytorch -c conda-forge &>> \logFiles\conda_env.log
# python3 -m pip install pynvml hic-straw &>> \logFiles\conda_env.log
# python3 -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-geometric
# conda deactivate

# cd ~/sequences_to_contact_maps
# conda create --name seq2contact_pytorch1.7.1_cuda10.2 -y
# source activate seq2contact_pytorch1.7.1_cuda10.2
# conda install -y pytorch=1.7.1 torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy -c pytorch -c conda-forge -c nvidia &>> \logFiles\conda_env.log
# python3 -m pip install pynvml hic-straw &>> \logFiles\conda_env.log
# conda env export > seq2contact_pytorch1.7.1_cuda10.2_environment.yml
# conda deactivate

# cd ~/sequences_to_contact_maps
# conda create --name seq2contact_pytorch1.8.1_cuda11.1 -y
# source activate seq2contact_pytorch1.8.1_cuda11.1
# conda install -y pytorch=1.8.1 torchvision cudatoolkit=11.1 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy -c pytorch -c conda-forge -c nvidia &>> \logFiles\conda_env.log
# python3 -m pip install pynvml hic-straw &>> \logFiles\conda_env.log
# conda env export > seq2contact_pytorch1.8.1_cuda11.1_environment.yml

envName=python3.8_pytorch1.8.1_cuda10.2
ofile=conda_env.log
cd ~/sequences_to_contact_maps
conda create --name $envName -y
source activate $envName
conda install -y python=3.8 pytorch=1.8.1 torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 -c pytorch -c conda-forge -c bioconda &>> \logFiles\conda_env.log
python3 -m pip install pynvml hic-straw &>> $ofile
python3 -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html &>> $ofile
python3 -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html &>> $ofile
python3 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html &>> $ofile
python3 -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html &>> $ofile
python3 -m pip install torch-geometric &>> $ofile
conda deactivate
