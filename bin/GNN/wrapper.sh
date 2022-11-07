#! /bin/bash


for i in 6 7 8 9 # midway3
# 5 midway3
# 0 1 2 3 4 # midway2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
