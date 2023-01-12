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
dirname="/project/depablo/erschultz/dataset_11_18_22-/project/depablo/erschultz/dataset_11_21_22"
m=1024
messagePassing='weighted_GAT'
preTransforms='constant-ContactDistance-GeneticDistance_norm-AdjPCs_12'
mlpModelID='none'
useEdgeAttr='true'
hiddenSizesList='8-8-8-8'
EncoderHiddenSizesList='64'
updateHiddenSizesList='1000-1000-64'
numHeads=8

yPreprocessing='sweeprand_log_inf'
yNorm='mean'
headArchitecture='bilinear_triu'
headArchitecture2='fc-fill'
headHiddenSizesList='1000-1000-1000-1000-1000-1000-1024'
rescale=2
useScratch='false'

k=12
useSignPlus='true'
batchSize=2
resumeTraining='true'

# 5th attempt signplus, more PCs

id=343
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch --use_scratch $useScratch
