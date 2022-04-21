#! /bin/bash


for i in 4 6 8 9
# 1 2 3 7
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
