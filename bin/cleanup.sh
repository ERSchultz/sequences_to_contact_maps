#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00

cd ~/scratch-midway2/dataset_09_02_21/samples

for i in 1 2 3 4 5 6 7 8 9
do
  echo $i
  rm -r "sample${i}"* &
done

wait

cd ~/scratch-midway2/

rm -r dataset_09_02_21
