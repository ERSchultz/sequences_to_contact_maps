#! /bin/bash

for i in 13 14 15 16
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
