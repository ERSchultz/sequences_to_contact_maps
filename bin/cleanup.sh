#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00



cd "/home/erschultz/scratch-midway2"

mv "contact_diffusion_kNN3" "/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02"


cd "/home/erschultz/scratch-midway2/dataset_04_26_22/samples"
for i in 1 2 3 4 5 6 7 8 9
do
  rm -r "sample${i}*" &
done

wait

cd "/home/erschultz/scratch-midway2"
rm -r dataset_04_26_22
