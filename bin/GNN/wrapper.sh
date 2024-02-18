#! /bin/bash

for i in 2 12
# 6 7 14 15
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
