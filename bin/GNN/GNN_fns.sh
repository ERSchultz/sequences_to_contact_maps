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
mlpModelID='none'

# preprocessing
yPreprocessing='diag'
yNorm='none'
yLogTransform='none'
maxDiagonal='none'
keepZeroEdges='false'
KR='false'
rescale='none'
meanFilt='none'
outputPreprocesing='none'

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
topK='none'
sparsifyThresholdUpper='none'
sparsifyThreshold='none'
loss='mse'
act='prelu'
innerAct='prelu'
headAct='prelu'
outAct='prelu'
trainingNorm='none'
headArchitecture='bilinear'
headArchitecture2='none'
headHiddenSizesList='none'
numHeads=1
gated='false'


# hyperparameters
nEpochs=100
batchSize=1
numWorkers=4
milestones='50'
gamma=0.1
splitSizes='none'
splitPercents='0.9-0.1-0.0'
randomSplit='true'
lr=1e-3
weightDecay=0.
dropout=0.


useScratch='true'
useScratchParallel='true'
verbose='false'
plotPredictions='true'
crop='none'
printParams='true'

train () {
  echo "id=${id}"
  python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --m $m --y_preprocessing $yPreprocessing --preprocessing_norm $yNorm --log_preprocessing $yLogTransform --kr $KR --rescale $rescale --mean_filt $meanFilt --output_preprocesing $outputPreprocesing --max_diagonal $maxDiagonal --keep_zero_edges $keepZeroEdges --message_passing $messagePassing --use_node_features $useNodeFeatures --use_edge_weights $useEdgeWeights --use_edge_attr $useEdgeAttr --hidden_sizes_list $hiddenSizesList  --encoder_hidden_sizes_list $EncoderHiddenSizesList --update_hidden_sizes_list $updateHiddenSizesList --transforms $transforms --pre_transforms $preTransforms --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --act $act --inner_act $innerAct --head_act $headAct --out_act $outAct --training_norm $trainingNorm --head_architecture $headArchitecture --head_architecture_2 $headArchitecture2 --head_hidden_sizes_list $headHiddenSizesList --num_heads $numHeads --gated $gated --n_epochs $nEpochs --lr $lr --weight_decay $weightDecay --dropout $dropout --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --split_sizes=$splitSizes --split_percents $splitPercents --random_split $randomSplit --verbose $verbose --scratch $scratch --use_scratch $useScratch --use_scratch_parallel $useScratchParallel --plot_predictions $plotPredictions --crop $crop --print_params $printParams --id $id --resume_training $resumeTraining --mlp_model_id $mlpModelID
}
