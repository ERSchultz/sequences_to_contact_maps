#! /bin/bash
#SBATCH --job-name=prep
#SBATCH --output=logFiles/prep.out
#SBATCH --time=6:00:00
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd '/home/erschultz/dataset_08_25_23/samples'
for i in {1..20}
do
  mkdir "sample${i}"
done
