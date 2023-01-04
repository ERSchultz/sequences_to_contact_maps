#! /bin/bash
#SBATCH --job-name=cleanup2
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup3.log
#SBATCH --time=2:00:00



dir='/project2/depablo/erschultz'
cd $dir

rm -r dataset_05_12_22_small

mkdir dataset_11_21_22_small
cd dataset_11_21_22_small
mkdir samples
cd samples
for i in {1..2400}
do
  mkdir "sample${i}"
done

for i in {1..2400}
do
  cd "${dir}/dataset_11_21_22/samples/sample${i}"
  rm -r *.png
  cp s.npy "${dir}/dataset_11_21_22_small/samples/sample${i}"
  cp config.json "${dir}/dataset_11_21_22_small/samples/sample${i}"
  cp y.npy "${dir}/dataset_11_21_22_small/samples/sample${i}"
  cp diag_chis_continuous.npy "${dir}/dataset_11_21_22_small/samples/sample${i}"
done


# cd $dir

# tar -czvf dataset_11_21_22.tar.gz dataset_11_21_22_small
