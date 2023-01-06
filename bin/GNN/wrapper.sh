#! /bin/bash

for i in 0 1
# 8 9 12 13 14 running
# 10 done midway2
# 5 11 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
