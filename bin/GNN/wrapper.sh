#! /bin/bash

for i in 13 14
# 9 10 11 12
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
