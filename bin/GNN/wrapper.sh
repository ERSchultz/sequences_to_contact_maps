#! /bin/bash

for i in 3 4
# 1 2 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
