#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00


dataset="/project2/depablo/erschultz/dataset_01_17_22/samples"
cd $dataset

rm -r sample19

for i in $( seq 1 4400 )
do
  cd "${dataset}/sample${i}"
  rm s.npy &
  rm e.npy &
  wait
done
