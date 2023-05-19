#! /bin/bash
#SBATCH --job-name=CGNNE6
#SBATCH --output=logFiles/ContactGNNEnergy6.out
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

rootName='ContactGNNEnergy6' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_05_15_23"
m=512
preTransforms='constant-ContactDistance-GeneticDistance_norm-AdjPCs_8'
hiddenSizesList='8-8-8-8'
EncoderHiddenSizesList='64'
updateHiddenSizesList='1000-1000-64'

outputPreprocesing='log'
headArchitecture='bilinear'
headArchitecture2='fc-fill_512'
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=2

k=8
useSignPlus='true'
batchSize=1

# sign_plus with log preprocessing


id=405
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
