#! /bin/bash
#SBATCH --job-name=CGNNE0
#SBATCH --output=logFiles/ContactGNNEnergy0.out
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
#SBATCH --dependency=afterok:8941921:8950390:8950389:8950388

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy0' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_09_28_23"
m=512
preTransforms='ContactDistance_corr-MeanContactDistance-MeanContactDistance_bonded-AdjPCs_10'
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
nEpochs=70
milestones='40'
maxSample=5000
inputLtoD='true'
inputLtoDMode='subtract'
loss='mse_log'

# subtract and mse_log
# 561 but lower lr

id=566
for lr in 1e-5
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
