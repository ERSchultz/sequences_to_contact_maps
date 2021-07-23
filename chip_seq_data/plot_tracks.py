import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp
import aggregate_peaks as agg_p

def loadData(dir, chr):
    accession_dirs = [osp.join(dir, file) for file in os.listdir(dir) if osp.isdir(osp.join(dir, file))]
    np_files = [osp.join(file, chr + '.npy') for file in accession_dirs]
    names = agg_p.get_names(dir, accession_dirs)

    first = True
    for name, file in zip(names, np_files):
        y = np.load(file)
        density = np.round(np.sum(y) / len(y), 3)
        if first:
            x = np.arange(0, len(y))
            first = False
        xy = np.stack([x, y])
        where = (y > 0) & (5000 < x) & (x  < 10000)

        plt.scatter(xy[0, where], xy[1, where], label = '{} {}'.format(name, density), alpha = 0.5)

    plt.legend()
    plt.show()


    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=osp.join('chip_seq_data','bedFiles'), help='directory of chip-seq data')
    parser.add_argument('--chr', default='1', help='which chromosome to plot')
    parser.add_argument('--res', default = 200, help='resolution for chip-seq data')
    args = parser.parse_args()

    loadData(args.dir, args.chr)


if __name__ == '__main__':
    main()
