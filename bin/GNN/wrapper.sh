#! /bin/bash


for i in 4 7 8 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
