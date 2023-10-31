#! /bin/bash

for i in 1 2 3 4 5 6 7 8 9
# done; 10 12, 588, 584,
# done: 9, 13, 14: 587, 585, 586,
do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
