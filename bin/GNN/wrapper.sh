#! /bin/bash

for i in 10 11
# 14 15 2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
