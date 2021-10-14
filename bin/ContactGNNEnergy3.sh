#! /bin/bash
#SBATCH --job-name=CGNNE3
#SBATCH --output=logFiles/ContactGNNEnergy3.out
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

dirname="/project2/depablo/erschultz/dataset_10_10_21"
deleteRoot='false'

modelType='ContactGNNEnergy'
rootName='ContactGNNEnergy2' # change to run multiple bash files at once
GNNMode='true'
outputMode='energy'

# architecture
k=2
m=1024
yPreprocessing='diag'
yNorm='none'
yLogTransform='True'
messagePassing='SignedConv'
useNodeFeatures='false'
useEdgeWeights='false'
hiddenSizesList='8-8-2'
transforms='none'
preTransforms='degree'
split_neg_pos_edges_for_feature_augmentation='true'
topK='none'
sparsifyThresholdUpper='none'
sparsifyThreshold=0.176
loss='mse'
act='prelu'
innerAct='prelu'
headAct='prelu'
outAct='prelu'
headArchitecture='concat-outer'
headHiddenSizesList='10-10-1'

# hyperparameters
nEpochs=1
batchSize=4
numWorkers=4
milestones='none'
gamma=0.1

useScratch='true'
verbose='true'
plotPredictions='true'
relabel_11_to_00='false'
crop='none'
printParams='true'

cd ~/sequences_to_contact_maps
source activate python3.8_pytorch1.8.1_cuda10.2

for lr in 1e-4
do
  python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --k $k --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --y_log_transform $yLogTransform --message_passing $messagePassing --use_node_features $useNodeFeatures --use_edge_weights $useEdgeWeights --hidden_sizes_list $hiddenSizesList --transforms $transforms --pre_transforms $preTransforms --split_neg_pos_edges_for_feature_augmentation $split_neg_pos_edges_for_feature_augmentation --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --act $act --inner_act $innerAct --head_act $headAct --out_act $outAct --head_architecture $headArchitecture --head_hidden_sizes_list $headHiddenSizesList --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --verbose $verbose --use_scratch $useScratch --plot_predictions $plotPredictions --relabel_11_to_00 $relabel_11_to_00 --crop $crop --print_params $printParams
done
python3 cleanDirectories.py --data_folder $dirname --root_name $rootName --use_scratch $useScratch
