#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00


cd "/project2/depablo/erschultz/michrom/project/chr_05"
rm -r sc_contacts

cd 'chr_05_02'
rm -r contact_diffusion_kNN2
