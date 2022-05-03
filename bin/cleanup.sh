#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00



cd "/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02/contact_diffusion"
rm -r iteration_2


cd "/home/erschultz/scratch-midway2/contact_diffusion"

for i in 0 2
do
  mv "iteration_${i}" "/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02/contact_diffusion"
done
