#! /bin/bash

for i in 4 6
# 12 0 1 2 7
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
