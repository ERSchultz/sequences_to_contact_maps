#! /bin/bash

for i in 17 18
# 13 14 15 16 23
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
