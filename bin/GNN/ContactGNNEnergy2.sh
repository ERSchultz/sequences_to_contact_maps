#! /bin/bash
#SBATCH --job-name=CGNNE2
#SBATCH --output=logFiles/ContactGNNEnergy2.out
#SBATCH --time=24:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=gpu2
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

rootName='ContactGNNEnergy2' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_11_18_22-/project2/depablo/erschultz/dataset_11_21_22"
m=1024
messagePassing='weighted_GAT'
preTransforms='constant-ContactDistance-GeneticDistance_norm-AdjPCs_8'
mlpModelID='none'
useEdgeAttr='true'
hiddenSizesList='8-8-8-8'
EncoderHiddenSizesList='64'
updateHiddenSizesList='1000-1000-64'
numHeads=8

outputPreprocesing='log'
yPreprocessing='sweeprand_log_inf'
yNorm='mean'
headArchitecture='bilinear_triu'
headArchitecture2='fc-fill'
headHiddenSizesList='1000-1000-1000-1000-1000-1000-1024'
rescale=2
useScratch='false'
resumeTraining='true'


k=8
useSignNet='true'
batchSize=2


# sign_net with log preprocessing

id=349
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch --use_scratch $useScratch
