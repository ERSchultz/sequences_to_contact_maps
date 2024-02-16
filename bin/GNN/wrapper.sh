#! /bin/bash

for i in 5 6 7 11 14 15
# 9 12 2 13
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
