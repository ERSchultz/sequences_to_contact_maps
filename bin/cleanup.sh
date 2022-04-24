#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00



cd "/home/erschultz/scratch-midway2/dataset_01_17_22/samples"
for i in $( seq 1 4 )
do
  echo $i
  rm "sample${i}*" &
done

wait

cd "/home/erschultz/scratch-midway2"
rm -r dataset_01_17_22

cd "/home/erschultz/scratch-midway2/dataset_09_02_21/samples"
for i in $( seq 1 4 )
do
  echo $i
  rm "sample${i}*" &
done

wait

cd "/home/erschultz/scratch-midway2"
rm -r dataset_09_02_21
