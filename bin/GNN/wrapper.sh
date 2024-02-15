#! /bin/bash

for i in 2 8 10
# 5 6 7 9 11 12
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
