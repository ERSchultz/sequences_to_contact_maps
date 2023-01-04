#! /bin/bash

for i in 9
# 0 1 2 4 5 8 10 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
