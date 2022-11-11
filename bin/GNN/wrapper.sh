#! /bin/bash

for i in 2 3
# 1 6 7 8 # midway2

#Thursday
# 6 7 8 # midway2
# 10 11 12 13 # midway3

# Monday
# 8 9 # midway2
# 5 midway3
# 0 2 4 # midway2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
