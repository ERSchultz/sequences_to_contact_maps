#! /bin/bash
#SBATCH --job-name=akita
#SBATCH --output=logFiles/akita.out
#SBATCH --time=20:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-gpu=12000
#SBATCH --account=pi-depablo

dirname="/project2/depablo/erschultz/dataset_04_18_21"
modelType='Akita'

# architecture
k=2
m=1024
yPreprocessing='diag'
yNorm='instance'
kernelWList='5-5-5'
hiddenSizesList='32-64-128'
dilationListTrunk='2-4-8-16-32-64-128-256-512'
bottleneck=32
dilationListHead='2-4-8-16-32-64-128-256-512'
outAct='sigmoid'
trainingNorm='batch'
downSampling='conv'


# hyperparameters
nEpochs=20
batchSize=4
numWorkers=4
milestones='none'
gamma=0.1
lr=1e-3


cd ~/sequences_to_contact_maps
source activate seq2contact_pytorch


python3 core_test_train.py --data_folder $dirname --model_type $modelType --k $k --m $m --y_preprocessing $yPreprocessing --y_norm $yNorm --kernel_w_list $kernelWList --hidden_sizes_list $hiddenSizesList --dilation_list_trunk $dilationListTrunk --bottleneck $bottleneck --dilation_list_head $dilationListHead --out_act $outAct --training_norm $trainingNorm --down_sampling $downSampling --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma

yNorm='none'
outAct='none'
python3 core_test_train.py --data_folder $dirname --model_type $modelType --k $k --m $m --y_preprocessing $yPreprocessing --y_norm $yNorm --kernel_w_list $kernelWList --hidden_sizes_list $hiddenSizesList --dilation_list_trunk $dilationListTrunk --bottleneck $bottleneck --dilation_list_head $dilationListHead --out_act $outAct --training_norm $trainingNorm --down_sampling $downSampling --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma
