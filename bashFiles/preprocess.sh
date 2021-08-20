#! /bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logFiles/preprocess.log
#SBATCH --time=01:00:00

input="/project2/depablo/skyhl/dataset_04_18_21"
output="/project2/depablo/erschultz/dataset_04_18_21"
numWorkers=20
k=2
n=1024
sampleSize=200
overwrite="False"

cd ~/sequences_to_contact_maps
source activate seq2contact_pytorch
python3 preprocess_data.py --input_folder $input --output_folder $output --num_workers $numWorkers --k $k --n $n --sample_size $sampleSize --overwrite ${overwrite}
