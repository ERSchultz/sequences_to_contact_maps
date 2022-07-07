#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=2:00:00



dataset="/project2/depablo/erschultz/dataset_05_18_22/samples"
cd $dataset

rm -r sample19

for i in 13 14 15
do
  cd "${datset}/sample${i}"
  rm -r GNN* &
  rm -r PCA* &
  wait
done
