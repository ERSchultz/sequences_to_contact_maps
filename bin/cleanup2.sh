#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=8:00:00



dir='/project2/depablo/erschultz'
cd $dir
# tar -czvf dataset_01_17_22.tar.gz dataset_01_17_22
# rm -r dataset_01_17_22
#
#
# for i in {1..2000}
# do
#   cd "${dir}/dataset_04_27_22/samples/sample${i}"
#   rm -r GNN* &
#   rm -r ground* &
#   rm -r k_means* &
#   rm -r PCA* &
#   rm e.npy &
#   rm y1000* &
#   rm y2500* &
#   rm y5000* &
#   rm -r data_out &
#   rm y_diag.npy &
#   rm s.npy &
#   wait
# done
# cd $dir
# tar -czvf dataset_04_27_22.tar.gz dataset_04_27_22
# rm -r dataset_04_27_22

for i in {1..2000}
do
  cd "${dir}/dataset_11_03_21/samples/sample${i}"
  rm -r GNN* &
  rm -r ground* &
  rm -r k_means* &
  rm -r PCA* &
  rm -r random* &
  rm e.npy &
  rm -r data_out &
  rm chis.tek &
  rm chis.npy &
  rm y_diag.npy &
  rm *.png &
  rm s.npy &
  wait
done
cd $dir
rm -r dataset_11_03_21.tar.gz
tar -czvf dataset_11_03_21.tar.gz dataset_11_03_21
rm -r dataset_11_03_21
