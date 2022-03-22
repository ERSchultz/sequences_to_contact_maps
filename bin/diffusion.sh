#! /bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=20000
#SBATCH --output=logFiles/diffusion.log
#SBATCH --time=12:00:00

dir="/project2/depablo/walt/michrom/project/chr_05/chr_05_01"
odir="/project2/depablo/erschultz/michrom/project/chr_05/chr_05_01"

cd ~/sequences_to_contact_maps
source activate python3.8_pytorch1.8.1_cuda10.2_2

python diffusion_analysis.py --dir $dir --odir $odir
