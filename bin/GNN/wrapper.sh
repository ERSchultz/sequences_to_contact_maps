#! /bin/bash

for i in 2 3
# 8 12 13 14 running
# 9 10 done midway2
# 0 1 5 11 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
