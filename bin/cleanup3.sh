#! /bin/bash
#SBATCH --job-name=cleanup3
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup3.log
#SBATCH --time=2:00:00



dir='/home/erschultz'
cd $dir

data_small="${dir}/dataset_02_04_23_small"
mkdir $data_small
cd $data_small
mkdir "samples"

for i in {201..282}
do
  mkdir "${data_small}/samples/sample${i}"
  mkdir "${data_small}/samples/sample${i}/optimize_grid_b_261_phi_0.01"
  mkdir "${data_small}/samples/sample${i}/optimize_grid_b_140_phi_0.03"
done

data_big="${dir}/dataset_02_04_23"
for i in {201..282}
do
  new="${data_small}/samples/sample${i}/optimize_grid_b_261_phi_0.01"
  cd "${data_big}/samples/sample${i}/optimize_grid_b_261_phi_0.01"
  cp -r y.npy $new
  cp -r grid_size.txt $new

  new="${data_small}/samples/sample${i}/optimize_grid_b_140_phi_0.03"
  cd "${data_big}/samples/sample${i}/optimize_grid_b_140_phi_0.03"
  cp -r y.npy $new
  cp -r grid_size.txt $new
done

cd $data_small
tar -czvf samples.tar.gz samples
