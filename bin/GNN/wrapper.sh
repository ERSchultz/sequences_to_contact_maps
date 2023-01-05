#! /bin/bash

for i in 4 8 9
# 0 1 2 5 completed
# 10 running
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
