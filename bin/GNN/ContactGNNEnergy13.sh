#! /bin/bash
#SBATCH --job-name=CGNNE13
#SBATCH --output=logFiles/ContactGNNEnergy13.out
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

rootName='ContactGNNEnergy13' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_01_02_23"
m=1024
messagePassing='weighted_GAT'
preTransforms='constant-ContactDistance-GeneticDistance_norm'
mlpModelID='none'
useEdgeAttr='true'
hiddenSizesList='8-8-8-8'
EncoderHiddenSizesList='1000-1000-64'
updateHiddenSizesList='1000-1000-64'
numHeads=8

yPreprocessing='sweeprand_log_inf'
yNorm='mean'
headArchitecture='bilinear_triu'
headArchitecture2='fc-fill_1024'
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=2
useScratch='false'


id=333
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch --use_scratch $useScratch
