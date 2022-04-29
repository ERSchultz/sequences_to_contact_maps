#! /bin/bash


for i in 2 4 5 6 7 8
# 1
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
