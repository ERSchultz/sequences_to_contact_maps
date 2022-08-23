#! /bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --output=logFiles/diffusion.log
#SBATCH --time=24:00:00

odir="/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017"
dir="${odir}/samples"
k=2
experimental='true'
jobs=10
metric='scc'
its=1


python3 /home/erschultz/sequences_to_contact_maps/diffusion_analysis.py --dir $dir --odir $odir --k $k --experimental $experimental --jobs $jobs --metric $metric --its $its
