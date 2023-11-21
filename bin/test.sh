#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=6:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu


# cd ~/sequences_to_contact_maps
# source activate python3.9_pytorch1.9_cuda10.2
#
# python3 test.py

dir="/home/erschultz/dataset_interp_test/samples_pool"
for i in {1..7}
do
  odir="${dir}/sample${i}"
  cd $odir
  rm import.log
  rm y.npy
  cd "Interpolation/zeros_mappability-0.7"
  mv "y_pool.npy" "${odir}/y.npy"
  mv "y_pool.png" "${odir}/y.png"
  cd $odir
  rm -r Interpolation
done
