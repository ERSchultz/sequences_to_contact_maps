#! /bin/bash

for i in 0 1 2 3 4 5
# 13 TODO

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
