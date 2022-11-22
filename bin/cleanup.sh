#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-gpu
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=4:00:00


dir='/project2/depablo/erschultz'

cd "${dir}/dataset_11_18_22"
rm -r ContactGNNEnergy0* &
rm -r ContactGNNEnergy1* &
rm -r ContactGNNEnergy2* &
rm -r ContactGNNEnergy3* &
rm -r ContactGNNEnergy4* &
rm -r ContactGNNEnergy5* &
# rm -r ContactGNNEnergy6* &
# rm -r ContactGNNEnergy7* &
# rm -r ContactGNNEnergy8* &
# rm -r ContactGNNEnergy9* &
wait
