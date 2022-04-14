#! /bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=depablo-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logFiles/preprocess.log
#SBATCH --time=04:00:00

input="/project2/depablo/erschultz/dataset_01_17_22"
output="/project2/depablo/erschultz/dataset_01_17_22"
numWorkers=16
m=1024
overwrite="false"
percentiles='none' # none skips percmeanDist_pathentiles
diagBatch='true'
sampleSize=200
splitPercents='none'

source activate python3.9_pytorch1.9_cuda10.2
cd ~/sequences_to_contact_maps
# python3 -m utils.preprocess_data --input_folder $input --output_folder $output --num_workers $numWorkers --m $m --overwrite $overwrite --percentiles $percentiles --diag_batch $diagBatch --sample_size $sampleSize --split_percents $splitPercents


python3 data_summary_plots.py
