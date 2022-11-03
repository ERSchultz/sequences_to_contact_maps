#! /bin/bash
#SBATCH --job-name=CGNNE2
#SBATCH --output=logFiles/ContactGNNEnergy2.out
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

rootName='ContactGNNEnergy2' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_09_30_22"
m=1024
messagePassing='weighted_GAT'
preTransforms='constant-degree_diag_split1-ContactDistance-GeneticDistance_norm'
mlpModelID='none'
useEdgeAttr='true'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='100-100-64'
updateHiddenSizesList='100-100-64'
numHeads=8

outputMode='energy_sym_diag'
yPreprocessing='log_inf'
yNorm='mean'
headArchitecture='bilinear'
headArchitecture2='fc-fill'
headHiddenSizesList='1000-1000-1000-1000-1000-1000-1024'

# like 225 but constant and split degree


id=228
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
