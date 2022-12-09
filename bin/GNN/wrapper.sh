#! /bin/bash

for i in 0 1 2 3 4 5 7 # midway2
# 6 8 running midway2
# 0 (290) 13 (277) done

do
  sbatch ~/sequences_to_contact_maps/bin/GNN/ContactGNNEnergy${i}.sh
done
