#! /bin/bash


for i in 0 1 2 3 4
# 10 11 12 13 running on midway3
# 7 8 9 running on midway3
# 5 6 cancelled these - want to rerun after re-implementing array to diag matrix
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
