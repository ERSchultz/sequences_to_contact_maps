#! /bin/bash

for i in 2 3 4
# 5 6 7 midway2
# 0 1 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
