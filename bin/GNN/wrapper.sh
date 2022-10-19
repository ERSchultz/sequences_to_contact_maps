#! /bin/bash


for i in 1 2 3
# 6 7
 # 4 5
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
