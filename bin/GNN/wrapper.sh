#! /bin/bash

for i in 3 5
# 1 2 4 6 7 8 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
