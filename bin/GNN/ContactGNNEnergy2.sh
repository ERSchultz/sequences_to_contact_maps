#! /bin/bash
#SBATCH --job-name=CGNNE2
#SBATCH --output=logFiles/ContactGNNEnergy2.out
#SBATCH --time=1-24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2

rootName='ContactGNNEnergy2' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_04_26_22"
m=2048
messagePassing='GAT'
preTransforms='degree-ContactDistance-GeneticDistance'
useEdgeAttr='true'
hiddenSizesList='64-64-64'
EncoderHiddenSizesList='100-100-64'
updateHiddenSizesList='100-100-64'
numHeads=1
batchSize=1

id=151
for lr in 1e-3
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
