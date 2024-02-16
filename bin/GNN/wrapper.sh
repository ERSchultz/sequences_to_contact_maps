#! /bin/bash

for i in 13
# 5 6 7 9 11 12 2
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
