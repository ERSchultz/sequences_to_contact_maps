#! /bin/bash

for i in 1 2
# 4 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
