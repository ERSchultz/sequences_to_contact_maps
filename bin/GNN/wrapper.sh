#! /bin/bash


for i in 1 2 3 4 5 6 7 8
# 3 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
