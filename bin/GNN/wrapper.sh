#! /bin/bash


for i in 5 6 7 10 11
# 1 3 running midway2

# Monday
# 8 9 # midway2
# 5 midway3
# 0 2 4 # midway2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
