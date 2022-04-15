#! /bin/bash
#SBATCH --job-name=CGNNE7
#SBATCH --output=logFiles/ContactGNNEnergy7.out
#SBATCH --time=1-24:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=3000

cd ~/sequences_to_contact_maps

source bin/GNN/GNN_fns.sh
source activate python3.9_pytorch1.9_cuda10.2

rootName='ContactGNNEnergy7' # change to run multiple bash files at once
dirname="/project2/depablo/erschultz/dataset_09_02_21"
splitSizes='none'
splitPercents='0.8-0.1-0.1'
preTransforms='degree-ContactDistance-GeneticDistance'
useEdgeWeights='false'
useEdgeAttr='true'
hiddenSizesList='32-32-32'
EncoderHiddenSizesList='100-100-64'
updateHiddenSizesList='100-100-64'
milestones='50'

id=139
for lr in 1e-3
do
  echo "id=${id}"
  python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --m $m --y_preprocessing ${yPreprocessing} --y_norm $yNorm --y_log_transform $yLogTransform --message_passing $messagePassing --use_node_features $useNodeFeatures --use_edge_weights $useEdgeWeights --use_edge_attr $useEdgeAttr --hidden_sizes_list $hiddenSizesList  --encoder_hidden_sizes_list $EncoderHiddenSizesList --update_hidden_sizes_list $updateHiddenSizesList --transforms $transforms --pre_transforms $preTransforms --split_edges_for_feature_augmentation $split_edges_for_feature_augmentation --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --act $act --inner_act $innerAct --head_act $headAct --out_act $outAct --head_architecture $headArchitecture --head_hidden_sizes_list $headHiddenSizesList --num_heads $numHeads --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --split_sizes $splitSizes --split_percents $splitPercents --verbose $verbose --scratch $scratch --use_scratch $useScratch --plot_predictions $plotPredictions --crop $crop --print_params $printParams --id $id --resume_training $resumeTraining
  id=$(( $id + 1 ))
done
python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
