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
jobs=19
metric='scc'
its=2
downSampling=1
plot='true'
preprocessingMode='sparsity_filter'


python3 /home/erschultz/sequences_to_contact_maps/diffusion_analysis.py --dir $dir --odir $odir --k $k --experimental $experimental --jobs $jobs --metric $metric --its $its --down_sampling $downSampling --plot $plot --preprocessing_mode $preprocessingMode
