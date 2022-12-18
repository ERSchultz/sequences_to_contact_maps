#! /bin/bash

for i in 0 1 2
# 10 11 12 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
