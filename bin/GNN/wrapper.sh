#! /bin/bash

for i in 19 20
# 0
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
