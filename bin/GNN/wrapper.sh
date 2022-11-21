#! /bin/bash

for i in 0 1 2
# 6 # midway2

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
