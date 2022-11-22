#! /bin/bash

for i in 4 5 # midway2 todo
# 0 1 2 3 6 # midway2

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
