#! /bin/bash

for i in 0 1
# 12 13 14 midway2
# 3 4 6 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
