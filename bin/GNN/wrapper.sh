#! /bin/bash

for i in 11
# 15 12
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
