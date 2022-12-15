#! /bin/bash

for i in 3 4 5
# 3 4 5 TODO midway3
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
