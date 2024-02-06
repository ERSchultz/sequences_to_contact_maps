#! /bin/bash
#SBATCH --job-name=CGNNE1
#SBATCH --output=logFiles/ContactGNNEnergy1.out
#SBATCH --time=1-24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu
#SBATCH --exclude=midway3-0372

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy1' # change to run multiple bash files at once
dirname="/project/depablo/erschultz/dataset_12_12_23_imr90"
m=512
preTransforms='ContactDistance-MeanContactDistance-AdjPCs_10'
hiddenSizesList='16-16-16-16'
updateHiddenSizesList='1000-1000-1000-1000-128'

outputPreprocesing='none'
headArchitecture='bilinear'
headArchitecture2="fc-fill_${m}"
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=2

act='leaky'
innerAct='leaky'
headAct='leaky'
outAct='leaky'

yNorm='mean_fill'
k=10
useSignPlus='true'
batchSize=1
nEpochs=60
milestones='40'
loss='mse_and_mse_log'
lossLambda1=1e-1


# 631 ablation
# mse_and_mse_log

id=641
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
