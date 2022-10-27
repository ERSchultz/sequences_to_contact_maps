#! /bin/bash


for i in 3 4
# 5 6  midway3
# 0 3 4 running on midway2
# 1 2 - done
# 10 11 12 13 running on midway3
# 7 8 9 running on midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
