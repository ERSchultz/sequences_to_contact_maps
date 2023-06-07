#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=4:00:00

dir='/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy'


for i in {1..500}
do
  cd $dir
  cd $i
  rm *upsampling.tar.gz
done
