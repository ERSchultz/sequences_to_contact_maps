#/bin/bash

dirname="/project2/depablo/erschultz/dataset_09_26_22"
# dirname="/home/erschultz/dataset_test2"
scratch='/scratch/midway2/erschultz'
# scratch="/home/erschultz/scratch"
deleteRoot='false'

modelType='ContactGNNEnergy'
GNNMode='true'
outputMode='energy_sym'
resumeTraining='false'
id='none'

# preprocessing
yPreprocessing='diag'
yNorm='none'
yLogTransform='10'
maxDiagonal='none'

# architecture
m=1024
messagePassing='SignedConv'
useNodeFeatures='false'
useEdgeWeights='false'
useEdgeAttr='false'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='100-100-16'
updateHiddenSizesList='100-100-16'
transforms='empty'
preTransforms='degree'
split_edges_for_feature_augmentation='true'
topK='none'
sparsifyThresholdUpper='none'
sparsifyThreshold=0.176
loss='mse'
act='prelu'
innerAct='prelu'
headAct='prelu'
outAct='prelu'
trainingNorm='none'
headArchitecture='bilinear'
headHiddenSizesList='none'
numHeads=1

# hyperparameters
nEpochs=100
batchSize=2
numWorkers=4
milestones='50'
gamma=0.1
splitSizes='-200-0'
splitPercents='none'
lr=1e-3

useScratch='true'
useScratchParallel='true'
verbose='false'
plotPredictions='true'
crop='none'
printParams='true'

train () {
  echo "id=${id}"
  python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --m $m --y_preprocessing ${yPreprocessing} --preprocessing_norm $yNorm --log_preprocessing $yLogTransform --max_diagonal $maxDiagonal --message_passing $messagePassing --use_node_features $useNodeFeatures --use_edge_weights $useEdgeWeights --use_edge_attr $useEdgeAttr --hidden_sizes_list $hiddenSizesList  --encoder_hidden_sizes_list $EncoderHiddenSizesList --update_hidden_sizes_list $updateHiddenSizesList --transforms $transforms --pre_transforms $preTransforms --split_edges_for_feature_augmentation $split_edges_for_feature_augmentation --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --act $act --inner_act $innerAct --head_act $headAct --out_act $outAct --training_norm $trainingNorm --head_architecture $headArchitecture --head_hidden_sizes_list $headHiddenSizesList --num_heads $numHeads --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --split_sizes=$splitSizes --split_percents $splitPercents --verbose $verbose --scratch $scratch --use_scratch $useScratch --use_scratch_parallel $useScratchParallel --plot_predictions $plotPredictions --crop $crop --print_params $printParams --id $id --resume_training $resumeTraining
}
