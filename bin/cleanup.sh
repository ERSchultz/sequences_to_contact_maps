#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00


cd "/project2/depablo/erschultz"

rm -r dataset_01_16_22 &
rm -r dataset_01_14_22 &
rm -r dataset_01_13_22 &

wait
