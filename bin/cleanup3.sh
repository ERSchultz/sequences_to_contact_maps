#! /bin/bash
#SBATCH --job-name=cleanup3
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup3.log
#SBATCH --time=2:00:00



dir='/project2/depablo/erschultz'
cd $dir

rm -r dataset_09_30_22_mini

cd dataset_11_21_22_small
cd samples
for i in {1..2400}
do
  cd "sample${i}"
  mkdir data_out
done

for i in {1001..1020}
do
  rm -r "sample${i}"
done

for i in {1..2400}
do
  cd "${dir}/dataset_11_21_22/samples/sample${i}/data_out"
  # rm -r *.png
  # cp s.npy "${dir}/dataset_11_21_22_small/samples/sample${i}"
  # cp config.json "${dir}/dataset_11_21_22_small/samples/sample${i}"
  # cp y.npy "${dir}/dataset_11_21_22_small/samples/sample${i}"
  # cp diag_chis_continuous.npy "${dir}/dataset_11_21_22_small/samples/sample${i}"
  cp contacts200000.txt "${dir}/dataset_11_21_22_small/samples/sample${i}/data_out"
  cp contacts300000.txt "${dir}/dataset_11_21_22_small/samples/sample${i}/data_out"
  cp contacts400000.txt "${dir}/dataset_11_21_22_small/samples/sample${i}/data_out"
  cp contacts500000.txt "${dir}/dataset_11_21_22_small/samples/sample${i}/data_out"
  cp contacts1000000.txt "${dir}/dataset_11_21_22_small/samples/sample${i}/data_out"
done


# cd $dir

# tar -czvf dataset_11_21_22.tar.gz dataset_11_21_22_small
