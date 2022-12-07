#! /bin/bash
#SBATCH --job-name=prep
#SBATCH --output=logFiles/prep.out
#SBATCH --time=6:00:00
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erschultz@uchicago.edu

# cd /project2/depablo/erschultz
# mkdir dataset_11_18_22_small
# cd dataset_11_18_22_small
# mkdir samples

# dir="/project2/depablo/erschultz/dataset_11_18_22_small/samples"
# for i in {1..2400}
# do
#   cd $dir
#   mkdir "sample${i}"
#   cd "sample${i}"
#   mkdir data_out
# done

dir="/project2/depablo/erschultz/dataset_11_18_22/samples"
for i in {1..2400}
do
  cd "$dir/sample${i}"
  # cp y.npy "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
  # cp s.npy "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
  # cp config.json "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
  # cp diag_chis_continuous.npy "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
  cd data_out
  for j in 1000000 200000 300000 400000 500000
  do
    cp "contacts${j}.txt" "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}/data_out"
  done
done

cd /project2/depablo/erschultz
tar -czf dataset_11_18_22_small.tar.gz dataset_11_18_22_small
