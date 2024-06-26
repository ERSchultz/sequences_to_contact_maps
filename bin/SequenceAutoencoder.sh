#! /bin/bash
#SBATCH --job-name=SequenceAutoencoder
#SBATCH --output=logFiles/SequenceAutoencoder.out
#SBATCH --time=20:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/../../../project2/depablo/erschultz/dataset_04_18_21"

modelType='SequenceFCAutoencoder'
autoencoderMode='true'
outputMode='sequence'

# architecture
k=2
m=1024
yPreprocessing='diag'
yNorm='instance'
hiddenSizesList='1024-128'
loss='BCE'
outAct='none'
parameterSharing='true'

# hyperparameters
nEpochs=100
batchSize=16
numWorkers=4
milestones='none'
gamma=0.1

useScratch='true'
verbose='false'
plotPredictions='true'

cd ~/sequences_to_contact_maps
source activate python3.8_pytorch1.8.1_cuda10.2


for lr in 1e-1 1e-2 1e-3
do
python3 core_test_train.py --data_folder $dirname --model_type $modelType --autoencoder_mode $autoencoderMode --output_mode $outputMode --k $k --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --hidden_sizes_list $hiddenSizesList --loss $loss --out_act $outAct --parameter_sharing $parameterSharing --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --use_scratch $useScratch --verbose $verbose --plot_predictions $plotPredictions
done
