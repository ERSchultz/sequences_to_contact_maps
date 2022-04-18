#! /bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=10
#SBATCH --output=logFiles/cleanup.log
#SBATCH --time=2:00:00

dir=/home/erschultz/sequences_to_contact_maps/results/Akita
for i in $( seq 3 8 )
do
  cd "dir/${i}$"
  rm *.pt
done

dir=/home/erschultz/sequences_to_contact_maps/results/DeepC
for i in $( seq 7 18 )
do
  cd "dir/${i}$"
  rm *.pt
done

dir=/home/erschultz/sequences_to_contact_maps/results/ContactGNN
for i in $( seq 1 228 )
do
  echo $i
  cd "dir/${i}$"
  rm *.pt
done

dir=/home/erschultz/sequences_to_contact_maps/results/GNNAutoencoder
for i in $( seq 24 56 )
do
  echo $i
  cd "dir/${i}$"
  rm *.pt
done

dir=/home/erschultz/sequences_to_contact_maps/results/GNNAutoencoder2
for i in $( seq 1 7 )
do
  echo $i
  cd "dir/${i}$"
  rm *.pt
done

dir=/home/erschultz/sequences_to_contact_maps/results/SequenceFCAutoencoder
for i in $( seq 3 8 )
do
  echo $i
  cd "dir/${i}$"
  rm *.pt
done

dir=/home/erschultz/sequences_to_contact_maps/results/UNet
for i in $( seq 1 39 )
do
  echo $i
  cd "dir/${i}$"
  rm *.pt
done
