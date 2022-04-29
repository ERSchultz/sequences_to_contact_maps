#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-gpu
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00




cd "/project2/depablo/erschultz/"

rm -r dataset_08_26_21 &
rm -r dataset_08_29_21 &

wait

cd "/home/erschultz/scratch-midway2/"

rm -r dataset_01_17_22 &
rm -r dataset_12_12_21 &

wait
