#! /bin/bash


for i in 1 3
 # 2 4 5 6 7 8
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
