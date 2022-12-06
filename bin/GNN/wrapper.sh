#! /bin/bash

for i in 6 13 0 1 2

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
