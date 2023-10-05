#! /bin/bash

for i in 10 11 12 13
  # pending: 19 22
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
