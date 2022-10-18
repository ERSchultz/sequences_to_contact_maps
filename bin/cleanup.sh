#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00


dir="/home/erschultz/scratch-midway2"
cd $dir
rm -r MLP*

dataset="${dir}/dataset_09_30_22/samples"
for i in $( seq 1 2520)
do
  sample="${dataset}/sample${i}"
  cd $sample
  rm y_log_diag.npy
  rm y_log.npy
  rm y_diag.npy
done
