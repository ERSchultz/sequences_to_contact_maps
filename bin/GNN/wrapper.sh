#! /bin/bash

for i in 5 6 7 8
# 0 todo
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
