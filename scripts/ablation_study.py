import csv
import os.path as osp
from collections import defaultdict

import numpy as np


def main():
    descr_dict = {427: 'original',
            430: r'predict $S$ (instead of $S^\dag$)',
            431: 'without H bonded',
            432: 'without ContactDistance',
            433: 'without meanConstact Distance',
            434: 'without genetic distance norm',
            435: 'without signconv (eigenvectors replaced with constant)',
            436: 'with gatv2conv instead of modified',
            437: 'with mean y_norm instead of mean_fill',
            438: 'without signconv (eigenvectors are naively included)',
            }
    loss_dict = defaultdict(lambda: None)

    for id in descr_dict.keys():
        gnn_dir = osp.join('/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy', str(id))
        log_file = osp.join(gnn_dir, 'out.log')
        if osp.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if line.startswith('Final val loss: '):
                        final_val_loss = np.round(float(line.split(':')[1].strip()), 3)
                        break
                else:
                    final_val_loss = None
                    print(f'Warning: loss not found for {log_file}')
            loss_dict[id] = final_val_loss

    print(loss_dict)


if __name__ == '__main__':
    main()
