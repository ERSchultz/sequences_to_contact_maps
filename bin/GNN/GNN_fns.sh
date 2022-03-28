#/bin/bash

dirname="/project2/depablo/erschultz/dataset_01_17_22"
# dirname="/home/erschultz/dataset_test2"
scratch='/scratch/midway2/erschultz'
# scratch="/home/erschultz/scratch"
deleteRoot='false'

modelType='ContactGNNEnergy'
GNNMode='true'
outputMode='energy'
resumeTraining='false'

# architecture
m=1024
yPreprocessing='diag'
yNorm='none'
yLogTransform='true'
messagePassing='SignedConv'
useNodeFeatures='false'
useEdgeWeights='false'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='100-100-16'
updateHiddenSizesList='100-100-16'
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
headArchitecture='outer'
headHiddenSizesList='100-100-1'

# hyperparameters
nEpochs=100
batchSize=2
numWorkers=4
milestones='none'
gamma=0.1
splitSizes='-200-0'
lr=1e-4

useScratch='true'
verbose='false'
plotPredictions='true'
crop='none'
printParams='true'
