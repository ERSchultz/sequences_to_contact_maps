#! /bin/bash

for i in 3 4 5
 # 6 7 8 midway2
# 3 4 5 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
