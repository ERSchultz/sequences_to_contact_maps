#! /bin/bash

for i in 18 10 11 12 13 14
  # 17
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
