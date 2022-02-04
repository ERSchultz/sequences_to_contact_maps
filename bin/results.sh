#! /bin/bash
#SBATCH --job-name=results
#SBATCH --output=logFiles/results.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug

dirname="/home/eric"
dataset="dataset_test"
method='none'
modelID='none'
k='4'
plot='true'

for sample in 91
do
  python ~/sequences_to_contact_maps/result_summary_plots.py --root $dirname --dataset $dataset --sample $sample --method $method --model_id $modelID --k $k --plot $plot
done
