#! /bin/bash
#SBATCH --job-name=CGNNE14
#SBATCH --output=logFiles/ContactGNNEnergy14.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=END
#SBATCH --mail-user=erschultz@uchicago.edu
#SBATCH --exclude=midway3-0372

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy14' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_08_25_23"
m=512
preTransforms='ContactDistance-MeanContactDistance-MeanContactDistance_bonded-AdjPCs_8'
hiddenSizesList='8-8-8-8'
updateHiddenSizesList='1000-1000-64'

outputPreprocesing='log'
headArchitecture='bilinear'
headArchitecture2="fc-fill_${m}"
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=2

act='leaky'
innerAct='leaky'
headAct='leaky'
outAct='leaky'

sweepChoices='2-3-4-5'
yNorm='mean_fill'
k=8
useSignPlus='true'
batchSize=1
nEpochs=80
milestones='40'
inputLtoD='true'

id=466
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
