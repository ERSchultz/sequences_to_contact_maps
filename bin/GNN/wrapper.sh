#! /bin/bash

for i in 19 22
  # pending: 15 17 19 18 9 21 14 23
	# done: 3, 20, 16
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
