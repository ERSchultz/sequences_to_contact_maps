#/bin/bash

useScratch='true'
splitSizes='-200-0'
splitPercents='none'
randomSplit='false'
gpus=1

train () {
  echo "id=${id}"
  python3 core_test_train.py --id $id --data_folder $dirname --scratch $scratch --model_type $modelType --output_mode $outputMode --preprocessing_norm $preprocessingNorm --log_preprocessing $logPreprocessing --y_zero_diag_count $yZeroDiagCount --m $m --hidden_sizes_list $hiddenSizesList --act $act --out_act $outAct --n_epochs $nEpochs --lr $lr --batch_size $batchSize --num_workers $numWorkers --milestones $milestones --gamma $gamma --split_sizes $splitSizes --split_percents $splitPercents --random_split $randomSplit --gpus $gpus --use_scratch $useScratch
}
