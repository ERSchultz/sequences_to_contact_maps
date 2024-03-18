#! bin/bash
dir="/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy"
for i in 690
do
	i_dir="${dir}/${i}"
	scp "${i_dir}/model.pt" erschultz@10.150.30.72:${i_dir}/model.pt
	scp "${i_dir}/model_early_stop.pt" erschultz@10.150.30.72:${i_dir}/model_early.pt
	# rm "${i_dir}/model.pt"
	# rm "${i_dir}/model_early_stop.pt"
done
