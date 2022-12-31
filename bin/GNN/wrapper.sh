#! /bin/bash

for i in 8 9 0 1
 # 2 3 4 10 11 midway2
# 5 6 7 midway2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
