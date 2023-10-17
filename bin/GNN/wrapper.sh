#! /bin/bash

for i in 10 11 12
	# 0 1 2 3 4 5 6 7 8
# 10 11 12
  # 17
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
