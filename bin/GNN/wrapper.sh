#! /bin/bash


for i in 1 2 3 5 7
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
