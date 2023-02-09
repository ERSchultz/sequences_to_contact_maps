#! /bin/bash
#SBATCH --job-name=cleanup3
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup3.log
#SBATCH --time=2:00:00



dir='/project/depablo/erschultz'
cd $dir

rm -r dataset*
