#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=2:00:00



cd "/project2/depablo/erschultz"

rm -r dataset_09_02_21 &
rm -r dataset_10_25_21 &
rm -r dataset_12_11_21 &
rm -r dataset_12_17_21 &
rm -r dataset_11_14_21 &
rm -r dataset_01_19_22 &
rm -r dataset_01_11_22 &
rm -r dataset_01_12_22 &

wait
