#! /bin/bash

for i in 2 3 4 5 # midway2
# 0 1 6 7 8 13 running midway2

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
