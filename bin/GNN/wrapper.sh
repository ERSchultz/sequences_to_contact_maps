#! /bin/bash

for i in 6 8 0 1
# 7 13 running

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
