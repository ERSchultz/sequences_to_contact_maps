#! /bin/bash


for i in 5 7
# 1
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
