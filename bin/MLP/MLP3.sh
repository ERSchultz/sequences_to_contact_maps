#! /bin/bash
#SBATCH --job-name=MLP3
#SBATCH --output=logFiles/MLP3.out
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
scratch="${scratch}/MLP3"


# preprocessing
preprocessingNorm='mean'
logPreprocessing='ln'
yZeroDiagCount=0
outputMode='diag_chi_continuous'

# architecture
m=1024
hiddenSizesList='2000-2000-2000-2000-2000-2000-1024'
act='prelu'
outAct='prelu'

# hyperparameters
nEpochs=100
batchSize=32
numWorkers=8
milestones='50'
gamma=0.1

# retry ln preprocessing with smaller lr

id=84
for lr in 1e-4 1e-5
do
  train
  id=$(( $id + 1 ))
done

python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --scratch $scratch --clean_scratch
