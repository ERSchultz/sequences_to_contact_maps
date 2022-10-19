#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup2.log
#SBATCH --time=2:00:00



dir="/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy"
cd $dir

for i in $( seq 1 169 )
do
  cd "${dir}/${i}"
  rm -r sample*
done
