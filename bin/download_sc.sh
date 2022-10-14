#! /bin/bash
#SBATCH --job-name=download
#SBATCH --output=logFiles/download.out
#SBATCH --time=1-24:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd /project2/depablo/erschultz
mkdir humanPFC
cd humanPFC
wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE130711&format=file
