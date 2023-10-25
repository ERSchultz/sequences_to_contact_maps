#! /bin/bash

for i in 9 10
	# 15 0
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
