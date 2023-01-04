#! /bin/bash
#SBATCH --job-name=CGNNE8
#SBATCH --output=logFiles/ContactGNNEnergy8.out
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

rootName='ContactGNNEnergy8' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_11_18_22-/project2/depablo/erschultz/dataset_11_21_22"
m=1024
messagePassing='weighted_GAT'
preTransforms='constant-ContactDistance-GeneticDistance_norm'
useEdgeAttr='true'
hiddenSizesList='8-8-8-8'
EncoderHiddenSizesList='none'
updateHiddenSizesList='1000-1000-64'
numHeads=8

yPreprocessing='sweeprand_log_inf'
yNorm='mean'
headArchitecture='bilinear_chi_triu'
headArchitecture2='fc-fill_1024'
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=2
useScratch='false'

useNodeFeatures='true'
k=8

# chi triu head arch - use node features

id=320
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
