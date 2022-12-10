#! /bin/bash

for i in 6 7 8
# 3 4 5 midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
