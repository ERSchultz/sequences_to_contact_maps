#! /bin/bash

for i in 2 3 4
# 0 1
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
