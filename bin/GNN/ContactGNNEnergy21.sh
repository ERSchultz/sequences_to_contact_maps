#! /bin/bash
#SBATCH --job-name=CGNNE21
#SBATCH --output=logFiles/ContactGNNEnergy21.out
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
#SBATCH --dependency=afterok:9107338:9107339:9107340:9107341:9107342


cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy21' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_09_25_23"
m=512
preTransforms='ContactDistance-MeanContactDistance-MeanContactDistance_bonded-AdjPCs_8'
hiddenSizesList='8-8-8-8'
updateHiddenSizesList='1000-1000-64'

outputPreprocesing='log'
headArchitecture='bilinear'
headArchitecture2="fc-fill_${m}"
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=1

act='leaky'
innerAct='leaky'
headAct='leaky'
outAct='leaky'

sweepChoices='5'
yNorm='mean_fill'
k=8
useSignPlus='true'
batchSize=1
nEpochs=80
milestones='40'

# ablation of 507 without rescale and with only 500k sweeps

id=513
for lr in 1e-5
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
