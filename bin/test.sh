#! /bin/bash
#SBATCH --job-name=test
#SBATCH --output=logFiles/test.out
#SBATCH --time=6:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

cd /home/erschultz/dataset_11_14_22/samples

# cd ~/sequences_to_contact_maps
# source activate python3.9_pytorch1.9_cuda10.2
#
# python3 test.py

for i in {101..105}
do
  echo $i
  odir="sample1${i}"
  mkdir -p $odir
  cp "sample${i}/y_interpolate_zscore_mappability.npy" "${odir}/y.npy"
  echo "copy of sample ${i} with zscore and mappability interpolation via mean along diagonal" > "${odir}/import.log"
done
