#! /bin/bash
#SBATCH --job-name=CGNNE10
#SBATCH --output=logFiles/ContactGNNEnergy10.out
#SBATCH --time=1-24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy10' # change to run multiple bash files at once
dirname="/project/depablo/erschultz/dataset_09_30_22"
m=1024
messagePassing='GAT'
preTransforms='degree-ContactDistance-GeneticDistance-DiagonalParameterDistance'
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

# get rid of split_edges_for_feature_augmentation
# with mean norm


id=205
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
