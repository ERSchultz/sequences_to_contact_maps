#! /bin/bash
#SBATCH --job-name=CGNNE2
#SBATCH --output=logFiles/ContactGNNEnergy2.out
#SBATCH --time=1-24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2

rootName='ContactGNNEnergy2' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_12_12_21"
splitSizes='none'
splitPercents='0.8-0.1-0.1'
headArchitecture='concat'
headHiddenSizesList='100-100-1'

id=147
for lr in 1e-3
do
  train
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
