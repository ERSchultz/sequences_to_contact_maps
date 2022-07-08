#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=2:00:00



dataset="/project2/depablo/erschultz/dataset_05_18_22/samples"
cd $dataset

rm -r sample19

for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
  cd "${datset}/sample${i}"
  rm -r GNN* &
  rm -r PCA* &
  rm -r ground* &
  wait
done
