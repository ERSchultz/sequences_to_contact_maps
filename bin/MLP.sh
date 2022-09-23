#! /bin/bash
#SBATCH --job-name=MLP
#SBATCH --output=logFiles/MLP.out
#SBATCH --time=01:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

cd ~/sequences_to_contact_maps

source activate python3.9_pytorch1.9

dirname="/home/erschultz/dataset_test_logistic"
scratch="/home/erschultz/scratch"
modelType='MLP'

# preprocessing
preprocessingNorm='mean'
logPreprocessing='none'
yZeroDiagCount=0
outputMode='diag_chi_continuous'

# architecture
m=1024
hiddenSizesList='2000-2000-2000-2000-2000-2000-1024'
act='prelu'
outAct='prelu'

# hyperparameters
nEpochs=70
batchSize=32
numWorkers=8
milestones='50'
gamma=0.1
splitSizes='none'
splitPercents='0.8-0.2-0'
randomSplit='true'
gpus=1

# other
useScratch='true'
for lr in 1e-4
do
  python3 core_test_train.py --data_folder $dirname --scratch $scratch --model_type $modelType --output_mode $outputMode --preprocessing_norm $preprocessingNorm --log_preprocessing $logPreprocessing --y_zero_diag_count $yZeroDiagCount --m $m --hidden_sizes_list $hiddenSizesList --act $act --out_act $outAct --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --split_sizes $splitSizes --split_percents $splitPercents --random_split $randomSplit --gpus $gpus --use_scratch $useScratch
  python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --scratch $scratch --clean_scratch 'true'
done
