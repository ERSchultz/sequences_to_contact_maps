#! /bin/bash
#SBATCH --job-name=CGNNE1
#SBATCH --output=logFiles/ContactGNNEnergy1.out
#SBATCH --time=1-24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu
#SBATCH --wait=22811370

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2

rootName='ContactGNNEnergy1' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_04_27_22"
m=1024
messagePassing='GAT'
preTransforms='degree-ContactDistance-GeneticDistance'
useEdgeAttr='true'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='100-100-64'
updateHiddenSizesList='100-100-64'
numHeads=8


id=164
lr=1e-4
for yPreprocessing in '5000_diag' '1000_diag' '2500_diag' 'diag'
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch > clean.log
