#! /bin/bash

for i in 3 17 19 20 16 18 9 21
	# 15 todo
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
