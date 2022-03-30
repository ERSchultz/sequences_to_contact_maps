#! /bin/bash
#SBATCH --job-name=diffusion2
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --output=logFiles/diffusion2.log
#SBATCH --time=24:00:00

dir="/project2/depablo/walt/michrom/project/chr_05/chr_05_02"
odir="/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02"
jobs=15

cd ~/sequences_to_contact_maps
source activate python3.8_pytorch1.8.1_cuda10.2_2
module load cmake

python3 diffusion_analysis.py --dir $dir --odir $odir --jobs $jobs
