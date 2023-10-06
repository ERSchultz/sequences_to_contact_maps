#! /bin/bash

for i in 11 12 13 22
# 10 11 12
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
