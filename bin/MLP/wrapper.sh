#! /bin/bash


for i in 1 2 3
do
  sbatch ~/sequences_to_contact_maps/bin/MLP/MLP${i}.sh
done
