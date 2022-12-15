#! /bin/bash

for i in 10 11 12 13 14 #midway3
# 0 1 2 6 7
# 3 4 5 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
