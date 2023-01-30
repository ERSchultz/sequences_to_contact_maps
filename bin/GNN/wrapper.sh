#! /bin/bash

for i in 4
# 1 2 3 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
