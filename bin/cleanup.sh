#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00



# cd "/project2/depablo/erschultz/dataset_04_26_22/samples"

# for i in $( seq 2001 4400)
# do
#   echo $i
#   rm -r "sample${i}"
# done


cd "/home/erschultz/scratch-midway2/dataset_04_26_22"

for i in $( seq 2001 4400)
do
  echo $i
  rm -r "sample${i}"
done


# cd "/project2/depablo/erschultz/"
#
# rm -r dataset_08_26_21 &
# rm -r dataset_08_29_21 &
#
# wait
