#! /bin/bash

for i in 1 2 10 8 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
