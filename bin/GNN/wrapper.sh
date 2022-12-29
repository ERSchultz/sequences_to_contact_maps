#! /bin/bash

for i in 10
 # 2 3 midway2
 # issue with 4, 8, 9
# 5 6 7 midway2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
