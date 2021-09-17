#! /bin/bash
#SBATCH --job-name=ContactGNN3
#SBATCH --output=logFiles/ContactGNN3.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/project2/depablo/erschultz/dataset_04_18_21"
deleteRoot='false'

modelType='ContactGNN'
rootName='ContactGNN3' # change to run multiple bash files at once
GNNMode='true'
outputMode='sequence'

# architecture
k=2
m=1024
yPreprocessing='diag'
yNorm='none'
yLogTransform='true'
messagePassing='SignedConv'
useNodeFeatures='false'
useEdgeWeights='false'
hiddenSizesList='16-2'
transforms='none'
preTransforms='none'
split_neg_pos_edges_for_feature_augmentation='true'
topK='none'
sparsifyThresholdUpper='none'
sparsifyThreshold=0.176
loss='BCE'
outAct='none'
headArchitecture='fc'
headHiddenSizesList='2'

# hyperparameters
nEpochs=5
batchSize=8
numWorkers=4
milestones='none'
gamma=0.1

useScratch='true'
verbose='false'
plotPredictions='true'
relabel_11_to_00='false'
printParams='true'

cd ~/sequences_to_contact_maps
source activate python3.8_pytorch1.8.1_cuda10.2

for preTransforms in 'degree'
do
  for lr in 1e-3
  do
    python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --k $k --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --y_log_transform $yLogTransform --message_passing $messagePassing --use_node_features $useNodeFeatures --use_edge_weights $useEdgeWeights --hidden_sizes_list $hiddenSizesList --transforms $transforms --pre_transforms $preTransforms --split_neg_pos_edges_for_feature_augmentation $split_neg_pos_edges_for_feature_augmentation --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --out_act $outAct --head_architecture $headArchitecture --head_hidden_sizes_list $headHiddenSizesList --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --verbose $verbose --use_scratch $useScratch --plot_predictions $plotPredictions --relabel_11_to_00 $relabel_11_to_00 --print_params $printParams
  done
  python3 cleanDirectories.py --data_folder $dirname --root_name $rootName --use_scratch $useScratch
done
