#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=4:00:00


cd ~
rm -r dataset_12_05_22

dir='/project2/depablo/erschultz'


cd "${dir}/dataset_11_21_22/samples"
for i in {2401..10000}
do
  rm -r "sample${i}"
done

# rm -r dataset_09_30_22_mini &
# rm -r dataset_09_30_22.tar.gz &
# wait

cd "${dir}/dataset_11_18_22"
rm -r ContactGNNEnergy0* &
rm -r ContactGNNEnergy1* &
rm -r ContactGNNEnergy2* &
rm -r ContactGNNEnergy3* &
rm -r ContactGNNEnergy4* &
rm -r ContactGNNEnergy5* &
rm -r ContactGNNEnergy6* &
rm -r ContactGNNEnergy7* &
rm -r ContactGNNEnergy8* &
rm -r ContactGNNEnergy9* &
rm -r ContactGNNEnergy10* &
rm -r ContactGNNEnergy11* &
rm -r ContactGNNEnergy12* &
rm -r ContactGNNEnergy13* &
wait

cd "${dir}/dataset_11_21_22"
rm -r ContactGNNEnergy0* &
rm -r ContactGNNEnergy1* &
rm -r ContactGNNEnergy2* &
rm -r ContactGNNEnergy3* &
rm -r ContactGNNEnergy4* &
rm -r ContactGNNEnergy5* &
rm -r ContactGNNEnergy6* &
rm -r ContactGNNEnergy7* &
rm -r ContactGNNEnergy8* &
rm -r ContactGNNEnergy9* &
rm -r ContactGNNEnergy10* &
rm -r ContactGNNEnergy11* &
rm -r ContactGNNEnergy12* &
rm -r ContactGNNEnergy13* &
wait
