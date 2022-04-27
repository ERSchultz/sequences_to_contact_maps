#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00



cd "/project2/depablo/erschultz"

rm -r dataset_08_26_21 &
rm -r dataset_08_29_21 &

wait
