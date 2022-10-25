#! /bin/bash


for i in 9 10 11 12 13
# 5 6 7 8 running on midway2
# 4 running on midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
