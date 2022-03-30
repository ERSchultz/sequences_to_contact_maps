#! /bin/bash

# cd ~/sequences_to_contact_maps
# conda create --name seq2contact_pytorch -y
# source activate seq2contact_pytorch
# conda install -y pytorch=1.8.1 torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy gcc-5 -c psi4 -c pytorch -c conda-forge &>> logFiles\conda_env.log
# python3 -m pip install pynvml hic-straw &>> \logFiles\conda_env.log
# python3 -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
# python3 -m pip install torch-geometric
# conda deactivate
#
# envName=python3.8_pytorch1.8.1_cuda11.1_nomkl
# ofile=conda_env11.1.log
# # conda create --name $envName -y
# source activate $envName
# conda install -y pytorch=1.8.1 pyg torchvision cudatoolkit=11.1 matplotlib imageio nomkl numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy -c pytorch -c conda-forge -c nvidia -c pyg &>> $ofile
# python3 -m pip install pynvml &>> $ofile
# conda env export > seq2contact_pytorch1.8.1_cuda11.1_nomkl_environment.yml
# conda deactivate

# envName=python3.8_pytorch1.8.1
# ofile=logFiles/conda_env.log
# conda create --name $envName -y
# source activate $envName
# conda install -y python=3.8 pytorch=1.8.1 pyg torchvision cudatoolkit=11.1 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg  &>> $ofile
# python3 -m pip install pynvml importmagic &>> $ofile
# conda deactivate

# envName=python3.8_pytorch1.8.1_cuda10.2_2
# ofile=logFiles/conda_env10.2.log
# conda create --name $envName -y
# source activate $envName
# conda install -y python=3.8 pytorch=1.8.1 pyg torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg &>> $ofile
# python3 -m pip install pynvml importmagic &>> $ofile
# conda deactivate

# envName=python2
# ofile=logFiles/conda_env2.log
# conda create --name $envName -y
# source activate $envName
# conda install -y python=2 numpy pandas scipy &>> $ofile
# conda deactivate

envName=python3.6
ofile=logFiles/conda_env3.6.log
conda create --name $envName -y
source activate $envName
conda install -y python=3.6.5 numpy=1.16.3 pytorch=1.1.0 cudatoolkit -c pytorch &>> $ofile
conda deactivate
