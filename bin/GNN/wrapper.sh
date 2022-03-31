#! /bin/bash

source ~/sequences_to_contact_maps/bin/GNN/GNN_fns.sh
local='false'

if [ $local = 'true' ]
then
  source activate python3.8_pytorch1.8.1
fi

for i in 1 2 3 4
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done

# python3 ~/sequences_to_contact_maps/utils/clean_directories.py --data_folder $dirname --clean_scratch $useScratch --scratch $scratch
