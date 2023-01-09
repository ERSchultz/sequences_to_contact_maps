#! /bin/bash

for i in 3 4 6
# 9 midway2
# 2 5 11 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
