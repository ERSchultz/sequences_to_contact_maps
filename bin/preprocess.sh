#! /bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logFiles/preprocess.log
#SBATCH --time=02:00:00

input="/project2/depablo/erschultz/dataset_08_29_21"
output="/project2/depablo/erschultz/dataset_08_29_21"
numWorkers=1
k=2
n=1024
sampleSize=200 # not used since use_batch_for_diag defaults to False
minSample=0
overwrite="true"
percentiles='none' # none skips percentiles

cd ~/sequences_to_contact_maps
source activate activate python3.8_pytorch1.8.1_cuda10.2
python3 preprocess_data.py --input_folder $input --output_folder $output --num_workers $numWorkers --k $k --n $n --sample_size $sampleSize --min_sample $minSample --overwrite $overwrite --percentiles $percentiles
