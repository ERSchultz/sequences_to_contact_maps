#! /bin/bash

for i in 7 8
# 9 10
# 0 todo
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
