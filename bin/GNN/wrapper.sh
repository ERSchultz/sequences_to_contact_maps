#! /bin/bash

for i in 10 9
# 4 8
# 0 1 2 5 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
