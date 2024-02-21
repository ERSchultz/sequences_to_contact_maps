#! bin/bash

for i in {400..630}
do
	cd "/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/${i}"
	rm *.tar.gz
done
