#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=4:00:00


dir="/home/erschultz/scratch-midway2"

cd "${dir}/dataset_09_30_22"
rm -r ContactGNNEnergy5 &
rm -r ContactGNNEnergy6 &
wait

dataset="${dir}/dataset_09_30_22/samples"
for i in $( seq 1 2520)
do
  sample="${dataset}/sample${i}"
  cd $sample
  rm y_log.npy
done

dir='/project2/depablo/erschultz'
dataset="${dir}/dataset_09_30_22/samples"
for i in $( seq 1 2520)
do
  sample="${dataset}/sample${i}"
  cd $sample
  rm y_log.npy
done
