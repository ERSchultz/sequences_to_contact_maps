#! /bin/bash
#SBATCH --job-name=results
#SBATCH --output=logFiles/results.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug

dirname="/home/eric/sequences_to_contact_maps"
dataset="dataset_11_14_21"
method='none'
modelID='none'
k='none'
plot='true'
linearModel='ols'
experimental='false'
overwrite='false'

for sample in 40
# 1 2 3 4 6 7 8 9 11 12 13 14 15 17 18 19 20 21 23 24
do
  python ~/sequences_to_contact_maps/result_summary_plots.py --root $dirname --dataset $dataset --sample $sample --method $method --model_id $modelID --k $k --plot $plot --experimental $experimental --linear_model $linearModel --overwrite $overwrite &
done

wait
