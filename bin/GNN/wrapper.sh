#! /bin/bash

for i in 2
# 0 1 midway2
# 2 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
