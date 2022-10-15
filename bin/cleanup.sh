#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00


dir="/home/erschultz/scratch-midway2"
dataset="${dir}/dataset_09_30_22/samples"
for i in $( seq 1 2520)
do
  sample="${dataset}/sample${i}"
  cd $sample
  rm y.npy
  rm y_diag.npy
  wait
done
