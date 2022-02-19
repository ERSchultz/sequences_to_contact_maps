#! /bin/bash
#SBATCH --job-name=results
#SBATCH --output=logFiles/results.out
#SBATCH --time=8:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

# dirname="/home/eric/sequences_to_contact_maps"
dirname='/project2/depablo/erschultz'
dataset="dataset_11_14_21"
method='none'
modelID='none'
k='none'
plot='true'
linearModel='ols'
experimental='false'
overwrite='false'

for sample in 40 1230 1718 1751 1761
# 1 2 3 4 6 7 8 9 11 12 13 14 15 17 18 19 20 21 23 24
do
  python ~/sequences_to_contact_maps/result_summary_plots.py --root $dirname --dataset $dataset --sample $sample --method $method --model_id $modelID --k $k --plot $plot --experimental $experimental --linear_model $linearModel --overwrite $overwrite &
done

wait
