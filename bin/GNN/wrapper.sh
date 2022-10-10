#! /bin/bash


for i in 2 6 8
# run 7 first
# 1
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
