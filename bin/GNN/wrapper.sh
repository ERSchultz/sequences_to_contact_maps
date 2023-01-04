#! /bin/bash

for i in 4 8
# 0 1 2 5 9 10 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
