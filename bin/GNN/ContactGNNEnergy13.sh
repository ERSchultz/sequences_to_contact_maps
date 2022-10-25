#! /bin/bash
#SBATCH --job-name=CGNNE9
#SBATCH --output=logFiles/ContactGNNEnergy13.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy13' # change to run multiple bash files at once
dirname="/project/depablo/erschultz/dataset_09_30_22"
m=1024
messagePassing='GAT'
preTransforms='degree_diag-ContactDistance-GeneticDistance_norm-DiagonalParameterDistance'
mlpModelID='none'
useEdgeAttr='true'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='100-100-64'
updateHiddenSizesList='100-100-64'
numHeads=8

split_edges_for_feature_augmentation='false'
outputMode='energy_sym'
yPreprocessing='log'
yNorm='mean'
scratch='/scratch/midway3/erschultz'

# try degree_diag with mean norm


id=208
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
