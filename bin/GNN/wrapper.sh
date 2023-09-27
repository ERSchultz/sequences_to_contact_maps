#! /bin/bash

for i in 20 21
# 19
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
