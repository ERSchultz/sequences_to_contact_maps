#! /bin/bash

for i in 5 6 7
# 0 1 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
