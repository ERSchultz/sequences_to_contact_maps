#! /bin/bash

for i in 15
# 9 10
# 13 14
# 11 12
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
