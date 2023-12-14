#! /bin/bash

for i in 2 3 5 6 7 9
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
