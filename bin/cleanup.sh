#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00


dir="/home/erschultz/scratch-midway2"
cd $dir

rm -r dataset_04*

dir="/project2/depablo/erschultz"
cd $dir

dataset="${dir}/${dataset_09_21_21/samples}"
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
  cd "${dataset}/sample${i}"
  rm -r k_means* &
  rm -r PCA* &
  rm -r nmf* &
  wait
done
