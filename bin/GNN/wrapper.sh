#! /bin/bash

for i in 4 6 7
# 0 todo
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
