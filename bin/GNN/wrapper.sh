#! /bin/bash


for i in 0 1 2 3 4
# 10 11 12 13 running on midway3
# 7 8 9 running on midway3
# 5 6  running on midway2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
