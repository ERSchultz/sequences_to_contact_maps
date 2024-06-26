#! /bin/bash
#SBATCH --job-name=CGNNE12
#SBATCH --output=logFiles/ContactGNNEnergy12.out
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
#SBATCH --dependency=afterany:9778039:9778040:9778041:9778042:9778043:9778044:9778045:9778046:9778047:9778048


cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2
source activate python3.9_pytorch1.9

rootName='ContactGNNEnergy12' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_12_06_23_max_ent_other_exp"
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
nEpochs=20
milestones='20'
loss='mse_log'
pretrainID=710
bondedPath='optimize_grid_b_200_v_8_spheroid_1.5'

# ablation 20k samples fine tune

id=717
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
