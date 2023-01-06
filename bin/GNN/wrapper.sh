#! /bin/bash

for i in 8
# 9 10 12 13 14 running
# 5 11 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
