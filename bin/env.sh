#! /bin/bash
#SBATCH --job-name=setup
#SBATCH --output=logFiles/setup.out
#SBATCH --time=1-24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

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
# conda activate $envName
# conda install -y python=3.8 pytorch=1.8.1 pyg torchvision cudatoolkit=11.1 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg  &>> $ofile
# python3 -m pip install pynvml importmagic &>> $ofile
# conda env export > logFiles/env_local.yml
# conda deactivate
#

# envName=python3.9_pytorch1.9
# ofile=logFiles/conda_env2.log
# # conda create --name $envName -y
# # conda activate $envName
# conda install -y python=3.9 pytorch=1.9 pyg torchvision cudatoolkit=11.1 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg  &>> $ofile
# python3 -m pip install pynvml importmagic &>> $ofile
# conda env export > logFiles/env_local2.yml
# conda deactivate

# midway3 env
envName=python3.9_pytorch1.9_cuda11.3
ofile=logFiles/conda_env_midway3.log
conda create --name $envName -y
conda activate $envName
conda install -y python=3.9 pytorch=1.9 pyg torchvision cudatoolkit=11.3 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg  &>> $ofile
python3 -m pip install pynvml importmagic hic-straw hicrep &>> $ofile
conda env export > logFiles/env_midway3.yml
conda deactivate


# envName=python3.9_pytorch1.11
# ofile=logFiles/conda_env2.log
# conda create --name $envName -y
# conda activate $envName
# conda install -y python=3.9 pytorch=1.11 pyg torchvision cudatoolkit=11.1 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg  &>> $ofile
# python3 -m pip install pynvml importmagic &>> $ofile
# conda env export > logFiles/env_local2.yml
# conda deactivate
#
# envName=python3.9_pytorch1.11_cuda10.2
# ofile=logFiles/conda_env2.log
# conda create --name $envName -y
# conda activate $envName
# conda install -y python=3.9 pytorch=1.11 pyg torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg  &>> $ofile
# python3 -m pip install pynvml importmagic &>> $ofile
# conda env export > logFiles/env_midway_39.yml
# conda deactivate

# envName=python3.9_pytorch1.9_cuda10.2
# conda create --name $envName -y
# conda activate $envName
# conda install -y python=3.9 pytorch=1.9 pyg torchvision cudatoolkit=10.2 matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg
# python3 -m pip install pynvml importmagic
# conda env export > logFiles/env_midway_39_19.yml
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

# envName=python3.6
# ofile=logFiles/conda_env3.6.log
# conda create --name $envName -y
# source activate $envName
# conda install -y python=3.6.5 numpy=1.16.3 pytorch=1.1.0 cudatoolkit -c pytorch &>> $ofile
# conda deactivate
