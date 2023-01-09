#! /bin/bash

for i in 12 13 14
# 9 midway2
# 3 4 6 midway3
# 2 5 11 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
