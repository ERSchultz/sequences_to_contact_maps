#! /bin/bash
#SBATCH --job-name=plotting
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logFiles/plotting.log
#SBATCH --time=06:00:00

dirname="/project2/depablo/erschultz/dataset_08_26_21"

# partition
gpus=1

# model
modelType='ContactGNNEnergy'
pretrained='true'

# other
plotPredictions='true'
plot='false'
useScratch='false'


cd ~/sequences_to_contact_maps
source activate python3.9_pytorch1.11_cuda10.2

for id in 23
do
  python3 plotting_functions.py --data_folder $dirname --model_type $modelType --id $id --gpus $gpus --pretrained $pretrained --plot_predictions $plotPredictions --use_scratch $useScratch
done

python3 cleanDirectories.py --data_folder $dirname --root_name $modelType --use_scratch $useScratch
