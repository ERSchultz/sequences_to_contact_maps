#! /bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=depablo-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logFiles/preprocess.log
#SBATCH --time=02:00:00

input="/home/erschultz/dataset_test2"
output="/home/erschultz/dataset_test2"
numWorkers=10
k=9
m=1024
overwrite="false"
percentiles='none' # none skips percmeanDist_pathentiles
diagBatch='true'
sampleSize=10

cd ~/sequences_to_contact_maps
python3 -m utils.preprocess_data --input_folder $input --output_folder $output --num_workers $numWorkers --k $k --m $m --overwrite $overwrite --percentiles $percentiles --diag_batch $diagBatch --sample_size $sampleSize
