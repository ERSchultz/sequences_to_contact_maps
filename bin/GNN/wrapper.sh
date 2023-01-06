#! /bin/bash

for i in 9
# 8 13 14 running
# 10 12 done midway2
# 0 1 2 3 5 11 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
