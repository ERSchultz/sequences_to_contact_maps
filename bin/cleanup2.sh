#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=2:00:00



cd "/project2/depablo/erschultz"

rm -r dataset_10_27_21 &
rm -r dataset_12_12_21 &
rm -r dataset_04_26_22 &
rm -r dataset_12_29_21 &
rm -r dataset_01_15_22 &


wait
