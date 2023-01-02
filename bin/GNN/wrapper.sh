#! /bin/bash

for i in 8
# 4 5
# 1 2 10 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
