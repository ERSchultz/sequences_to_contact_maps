#! /bin/bash
#SBATCH --job-name=deepC
#SBATCH --output=logFiles/deepC.out
#SBATCH --time=10:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/../../../project2/depablo/erschultz/dataset_04_18_21"
modelType='DeepC'

# architecture
k=2
m=1024
yPreprocessing='diag'
yNorm='batch'
kernelWList='5-5-5'
hiddenSizesList='32-64-128'
dilationList='2-4-8-16-32-64-128-256-512'
outAct='sigmoid'
trainingNorm='batch'

# hyperparameters
nEpochs=20
batchSize=32
numWorkers=4
milestones=None
gamma=0.1

cd ~/sequences_to_contact_maps
source activate seq2contact_pytorch


for lr in 1e-2 1e-3 1e-4
do
python3 core_test_train.py --data_folder $dirname --model_type $modelType --k $k --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --kernel_w_list $kernelWList --hidden_sizes_list $hiddenSizesList --dilation_list $dilationList --out_act $outAct --training_norm $trainingNorm --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma
done
