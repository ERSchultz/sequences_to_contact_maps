#! /bin/bash

for i in 0 1 2 3 4 5 6 7 8
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
