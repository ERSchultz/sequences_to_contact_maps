#! /bin/bash

for i in 9 11 12 13 14 15
# 7 8 9 10
# 4 6 0
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
