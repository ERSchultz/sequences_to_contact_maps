#! /bin/bash

for i in 7
# 2 3 4 5 6 # midway3
# 0 midway2
# 1 paused
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
