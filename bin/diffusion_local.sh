#! /bin/bash

odir="/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017"
dir="${odir}/samples"
k=8
experimental='true'
jobs=14
metric='scc'
its=1
downSampling=1
plot='true'
preprocessingMode='sparsity_filter'
inputFileType='mcool'
chroms='all'
resolution=500000

python3 /home/erschultz/sequences_to_contact_maps/diffusion_analysis.py --dir $dir --odir $odir --k $k --experimental $experimental --jobs $jobs --metric $metric --its $its --down_sampling $downSampling --plot $plot --preprocessing_mode $preprocessingMode --input_file_type $inputFileType --chroms $chroms --resolution $resolution
