#! /bin/bash
#SBATCH --job-name=UNet
#SBATCH --output=logFiles/UNet2.out
#SBATCH --time=05:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/../../../project2/depablo/erschultz/dataset_04_18_21"
modelType='UNet'

# preprocessing
toxx='true'
xReshape='false'

# architecture
k=2
m=1024
yPreprocessing='diag_instance'
yNorm='instance'
minSubtraction='False'
nf=8
loss='mse'
outAct='sigmoid'
trainingNorm='batch'

# hyperparameters
nEpochs=15
batchSize=32
numWorkers=4
milestones='5-10'
gamma=0.1

cd ~/sequences_to_contact_maps
source activate seq2contact_pytorch

for lr in 1e-1
do
  python3 core_test_train.py --data_folder $dirname --model_type $modelType --toxx $toxx --x_reshape $xReshape --k $k --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --min_subtraction $minSubtraction --nf $nf --loss $loss --out_act $outAct --training_norm $trainingNorm --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma
done
