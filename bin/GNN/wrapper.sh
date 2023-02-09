#! /bin/bash

for i in 1 3
# 2 running
# 4 5 TODO
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
