#! /bin/bash

for i in 0 1 2 
# 5 6 7 8 9 10 11 12
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
