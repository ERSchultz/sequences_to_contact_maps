#! /bin/bash

for i in 0 1
# 6 7 8 13 running

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
