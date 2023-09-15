#! /bin/bash

for i in 11 12
# 15
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
