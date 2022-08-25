#! /bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=depablo-gpu
#SBATCH --ntasks=16
#SBATCH --output=logFiles/diffusion.log
#SBATCH --time=24:00:00

odir="/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017"
scratch='/scratch/midway2/erschultz'
dir="${odir}/samples"
k=2
experimental='true'
jobs=15
metric='scc'
its=2
downSampling=10
plot='true'
preprocessingMode='sparsity_filter'

conda activate python3.9_pytorch1.9_cuda10.2

python3 /home/erschultz/sequences_to_contact_maps/diffusion_analysis.py --dir $dir --scratch $scratch --odir $odir --k $k --experimental $experimental --jobs $jobs --metric $metric --its $its --down_sampling $downSampling --plot $plot --preprocessing_mode $preprocessingMode
