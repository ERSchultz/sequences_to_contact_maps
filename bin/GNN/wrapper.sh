#! /bin/bash

for i in 9
# 2 5 11 midway3
# 8 12 13 14 midway2 done
# 0 1 3 midway3 done
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
