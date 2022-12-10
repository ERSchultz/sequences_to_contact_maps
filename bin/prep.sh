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

# cd /project2/depablo/erschultz
# rm -r dataset_11_18_22_small.tar.gz

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
#
# dir="/project2/depablo/erschultz/dataset_11_18_22/samples"
# for i in {1..2400}
# do
#   cd "$dir/sample${i}"
#   # cp y.npy "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
#   # cp s.npy "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
#   # cp config.json "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
#   # cp diag_chis_continuous.npy "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}"
#   cd data_out
#   for j in 1000000 200000 300000 400000 500000
#   do
#     cp "contacts${j}.txt" "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}/data_out"
#   done
# done

# mkdir dataset_11_18_22_small2
# cd dataset_11_18_22_small2
# mkdir samples
#
# cd /project2/depablo/erschultz
# for i in {1200..2400}
# do
#   mv "/project2/depablo/erschultz/dataset_11_18_22_small/samples/sample${i}" "/project2/depablo/erschultz/dataset_11_18_22_small2/samples/sample${i}"
# done
#
# tar -czf dataset_11_18_22_small.tar.gz dataset_11_18_22_small &
# tar -czf dataset_11_18_22_small2.tar.gz dataset_11_18_22_small2 &
# wait


cd /project2/depablo/erschultz
tar -xzf dataset_11_18_22_small.tar.gz &
tar -xzf dataset_11_18_22_small2.tar.gz &
wait
