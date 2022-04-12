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
n=1024
sampleSize=200 # not used since use_batch_for_diag defaults to False
minSample=0
overwrite="false"
percentiles='none' # none skips percentiles

cd ~/sequences_to_contact_maps
python3 -m utils.preprocess_data --input_folder $input --output_folder $output --num_workers $numWorkers --k $k --n $n --sample_size $sampleSize --min_sample $minSample --overwrite $overwrite --percentiles $percentiles
