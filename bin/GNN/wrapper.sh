#! /bin/bash


for i in 2 8
# 1 3 - TODO midway3
# 5 6 7 - running (poorly)
 # 4 - starting
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
