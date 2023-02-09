#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=4:00:00

dir='/project2/depablo/erschultz'

rm -r *plaid_cutoff
