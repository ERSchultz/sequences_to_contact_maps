#! /bin/bash

for i in 6 7 8 9 10 11
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
