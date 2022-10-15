#! /bin/bash


for i in 8 9
# 6 still running
# original 7 has incorrect preprocessing - not sure if it is interesting or not
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
