#! /bin/bash

for i in 2 5 11
# 8 9 13 14 running
# 12 done midway2
# 0 1 3 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
