#! /bin/bash
#SBATCH --job-name=results
#SBATCH --output=logFiles/results.out
#SBATCH --time=0:30:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2000
#SBATCH --qos=depablo-debug

dirname="/home/eric/sequences_to_contact_maps"
dataset="dataset_01_15_22"
sample=40
method='GNN'
modelID=70
k='4'


python ~/sequences_to_contact_maps/result_summary_plots.py --root $dirname --dataset $dataset --sample $sample --method $method --model_id $modelID --k $k
