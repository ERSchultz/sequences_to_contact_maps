#! /bin/bash

source ~/sequences_to_contact_maps/bin/GNN/GNN_fns.sh
local='false'

if [ $local = 'true' ]
then
  source activate python3.8_pytorch1.8.1
else
  source activate python3.8_pytorch1.8.1_cuda10.2_2
fi

cd ~/sequences_to_contact_maps

for i in 1 2 3 4
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh &
  sleep 5 # hacky solution to prevent jobs from picking the same ID
done
wait

python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --clean_scratch $useScratch --scratch $scratch
