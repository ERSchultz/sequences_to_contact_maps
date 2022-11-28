#! /bin/bash

for i in 10 11 12 13
# 0 1 2 3 7 8 9

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
