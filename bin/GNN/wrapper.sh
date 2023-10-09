#! /bin/bash

for i in 14 16
# 10 11 12 13 14 15
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
