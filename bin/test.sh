#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug

dirname="/../../../project2/depablo/erschultz/dataset_04_18_21"


cd ~/sequences_to_contact_maps
source activate python3.9_pytorch1.11_cuda10.2

python3 test.py
