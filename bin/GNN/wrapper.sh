#! /bin/bash

for i in 4 5 6 7
# 12 0 1 2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
