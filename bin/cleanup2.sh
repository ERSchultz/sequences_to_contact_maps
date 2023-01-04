#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=2:00:00



dir='/project2/depablo/erschultz'
cd $dir
tar -czvf dataset_01_17_22.tar.gz dataset_01_17_22
rm -r dataset_01_17_22


for i in {1..2000}
do
  cd "${dir}/dataset_04_27_22/samples/sample${i}"
  rm -r GNN* &
  rm -r ground* &
  rm -r k_means* &
  rm -r PCA* &
  rm e.npy &
  rm y1000* &
  rm y2500* &
  rm y5000* &
  rm -r data_out &
  rm y_diag.npy &
  rm s.npy &
  wait
done
cd $dir
tar -czvf dataset_04_27_22.tar.gz dataset_04_27_22
rm -r dataset_04_27_22

for i in {1..2000}
do
  cd "${dir}/dataset_05_12_22/samples/sample${i}"
  rm -r GNN* &
  rm -r ground* &
  rm -r k_means* &
  rm -r PCA* &
  rm e.npy &
  rm -r data_out &
  rm chis.tek &
  rm y_diag.npy &
  rm s.npy &
  wait
done
cd $dir
tar -czvf dataset_05_12_22.tar.gz dataset_05_12_22
rm -r dataset_05_12_22

tar -czvf dataset_05_12_22.tar.gz dataset_05_12_22
rm -r dataset_05_12_22
