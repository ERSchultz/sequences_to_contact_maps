#! /bin/bash
#SBATCH --job-name=seq2energy
#SBATCH --partition=depablo-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logFiles/seq2energy.log
#SBATCH --time=06:00:00

dataset="dataset_11_03_21"
# partition
gpus=1

# model
modelType='ContactGNNEnergy'
pretrained='true'
local='true'
useScratch='false'
root='None'
deleteRoot='true'

cd ~/sequences_to_contact_maps
if [ $local = 'true' ]
then
  source activate python3.8_pytorch1.8.1_cuda11.1
  dirname="/home/eric/sequences_to_contact_maps/${dataset}"
else
  source activate python3.8_pytorch1.8.1_cuda10.2
  dirname="/project2/depablo/erschultz/${dataset}"
fi


for id in 42
do
  python3 train_seq2energy.py --data_folder $dirname --model_type $modelType --id $id --gpus $gpus --pretrained $pretrained --use_scratch $useScratch --root_name $root --delete_root $deleteRoot
done
