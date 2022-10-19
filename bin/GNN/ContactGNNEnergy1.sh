#! /bin/bash
#SBATCH --job-name=CGNNE1
#SBATCH --output=logFiles/ContactGNNEnergy1.out
#SBATCH --time=1-24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2

rootName='ContactGNNEnergy1' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_09_30_22"
m=1024
messagePassing='weighted_GAT'
preTransforms='degree-ContactDistance-GeneticDistance'
mlpModelID='none'
useEdgeAttr='true'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='100-100-64'
updateHiddenSizesList='100-100-64'
numHeads=8

outputMode='energy_sym_diag'
yPreprocessing='log'
yLogTransform='none'
sparsifyThreshold='none'
yNorm='mean'
headArchitecture2='concat'
headHiddenSizesList='100-100-100-100-100-100'
scratch='/scratch/midway3/erschultz'


# Run on midway3

id=195
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch > clean.log
