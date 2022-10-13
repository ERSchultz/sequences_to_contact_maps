#! /bin/bash
#SBATCH --job-name=MLP
#SBATCH --output=logFiles/MLP2.out
#SBATCH --time=06:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=2000

cd ~/sequences_to_contact_maps

source bin/MLP/MLP_fns.sh
source activate python3.9_pytorch1.9
source activate python3.9_pytorch1.9_cuda10.2

dirname="/project2/depablo/erschultz/dataset_09_30_22"
scratch='/scratch/midway2/erschultz'
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
splitSizes='900-200-0'

# train on subset


id=82
for lr in 1e-3 1e-4
do
  train
  id=$(( $id + 1 ))
done
