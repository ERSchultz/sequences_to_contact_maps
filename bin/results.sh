#! /bin/bash
#SBATCH --job-name=results
#SBATCH --output=logFiles/results.out
#SBATCH --time=8:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000

dirname="/home/erschultz"
# dirname="/home/erschultz"
# dirname='/project2/depablo/erschultz'
dataset="dataset_01_26_23"
sample='none'
sampleFolder='none'
method='none'
modelID='none'
k='none'
plot='true'
experimental='false'
overwrite='false'
robust='false'
scale='false'
svd='false'

source activate python3.9_pytorch1.9

for i in 202
# 4 6 8
# 1 2 3 4 6 7 8 9 11 12 13 14 15 17 18 19 20 21 23 24
do
  sampleFolder="${dirname}/${dataset}/samples/sample${i}"
  python ~/sequences_to_contact_maps/result_summary_plots.py --root $dirname --dataset $dataset --sample $sample --sample_folder $sampleFolder --method $method --model_id $modelID --k $k --plot $plot --experimental $experimental --overwrite $overwrite --robust $robust --scale $scale --svd $svd &
done

wait
