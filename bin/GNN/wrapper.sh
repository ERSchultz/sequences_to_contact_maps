#! /bin/bash

for i in 15 17
# 11 12 13 16 19
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
