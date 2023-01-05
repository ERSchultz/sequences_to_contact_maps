#! /bin/bash

for i in 12 13 14
# 11 midway3
# 0 1 2 5 completed
# 4 8 9 10 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
