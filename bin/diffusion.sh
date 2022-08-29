#! /bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=depablo-ivyb
#SBATCH --ntasks=20
#SBATCH --output=logFiles/diffusion.log
#SBATCH --time=24:00:00

odir="/project2/depablo/erschultz/single_cell_nagano_2017"
scratch='/scratch/midway2/erschultz'
dir="${odir}/samples"
k=2
experimental='true'
jobs=20
metric='scc'
its=1
downSampling=1
plot='true'
preprocessingMode='sparsity_filter'
inputFileType='mcool'
chroms='all'

conda activate python3.9_pytorch1.9_cuda10.2

python3 /home/erschultz/sequences_to_contact_maps/diffusion_analysis.py --dir $dir --odir $odir --k $k --experimental $experimental --jobs $jobs --metric $metric --its $its --down_sampling $downSampling --plot $plot --preprocessing_mode $preprocessingMode --input_file_type $inputFileType --chroms $chroms
