#! /bin/bash

for i in 13 0 1 2 3 4 5
# 10 11 12 13
# 7 8 9

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
