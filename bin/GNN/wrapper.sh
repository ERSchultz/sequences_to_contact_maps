#! /bin/bash

for i in 0 1 2 4 8 10
# 5 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
