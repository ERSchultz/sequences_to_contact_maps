#! /bin/bash
#SBATCH --job-name=simpleEpiNet
#SBATCH --output=logFiles/simpleEpiNet.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/../../../project2/depablo/erschultz/dataset_04_18_21"
modelType='SimpleEpiNet'

# architecture
k=2
m=1024
yPreprocessing='diag'
kernelWList='3-5-5'
hiddenSizesList='10-20-30'

# hyperparameters
nEpochs=2
lr=0.1
batchSize=32
numWorkers=4
milestones='5-10'
gamma=0.1

cd ~/sequences_to_contact_maps
source activate seq2contact_pytorch
python3 main.py --data_folder $dirname --model_type $modelType --k $k --m $m --y_preprocessing $yPreprocessing --kernel_w_list $kernelWList --hidden_sizes_list $hiddenSizesList --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma
