#! /bin/bash
#SBATCH --job-name=ContactGNN
#SBATCH --output=logFiles/ContactGNN.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/../../../project2/depablo/erschultz/dataset_04_18_21"
deleteRoot='false'

modelType='ContactGNN'
rootName='ContactGNN' # change to run multiple bash files at once
GNNMode='true'
outputMode='sequence'

# architecture
k=2
m=1024
yPreprocessing='diag'
yNorm='none'
yLogTransform='false'
messagePassing='GCN'
useNodeFeatures='false'
useEdgeWeights='true'
hiddenSizesList='16-2'
transforms='constant'
preTransforms='none'
topK='none'
sparsifyThresholdUpper='none'
sparsifyThreshold='none'
loss='BCE'
outAct='none'
headArchitecture='none'
headHiddenSizesList='none'

# hyperparameters
nEpochs=20
batchSize=8
numWorkers=4
milestones='none'
gamma=0.1

useScratch='true'
verbose='false'
plotPredictions='true'
relabel_11_to_00='true'

cd ~/sequences_to_contact_maps
source activate python3.8_pytorch1.8.1_cuda10.2

for sparsifyThreshold in 0.5 1.0 1.5
do
  for lr in 1e-1 1e-2 1e-3
  do
    python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --k $k --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --y_log_transform $yLogTransform --message_passing $messagePassing --use_node_features $useNodeFeatures --use_edge_weights $useEdgeWeights --hidden_sizes_list $hiddenSizesList --transforms $transforms --pre_transforms $preTransforms --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --out_act $outAct --head_architecture $headArchitecture --head_hidden_sizes_list $headHiddenSizesList --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --verbose $verbose --use_scratch $useScratch --plot_predictions $plotPredictions --relabel_11_to_00 $relabel_11_to_00
  done
  python3 cleanDirectories.py --data_folder $dirname --root_name $rootName --use_scratch $useScratch
done
