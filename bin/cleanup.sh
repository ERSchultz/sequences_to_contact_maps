#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00

cd ~/scratch-midway2/dataset_01_17_22/ContactGNNEnergy1

rm graph_1* &
rm graph_2* &
rm graph_3* &
rm graph_4* &

wait

cd ..

rm -r ContactGNNEnergy1
