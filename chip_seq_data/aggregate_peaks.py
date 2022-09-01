import argparse
import csv
import decimal
import os
import os.path as osp

import numpy as np
import pandas as pd
from utils import *


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=osp.join('chip_seq_data','narrow_peaks'), help='directory of chip-seq data')
    parser.add_argument('--res', default=200, help='resolution for chip-seq data')
    parser.add_argument('--file_type', default='.bed', help='file format')
    parser.add_argument('--cell_line', default='HTC116', help='cell line')
    args = parser.parse_args()

    return args

def aggregate_peaks(ifile, resolution):
    '''
    Uses approach from Zhou et al (DeepSEA) to aggregate peaks from Chip-seq data.
    Reference:
        https://doi.org/10.1038/nmeth.3547
    Inputs:
        ifile: file location of single chip-seq narrow peak track in bed format
        resolution: resolution of data
    Outputs:
        x: dictionary mapping chr (int) : aggregated peaks (numpy array)
    '''
    dir = ifile.split('.')[0]
    if not osp.exists(dir):
        os.mkdir(dir, mode = 0o755)

    x = {}
    for i_c in CHROMS:
        x[i_c] = np.zeros(int(CHROM_LENGTHS[i_c] / resolution))

    with open(ifile, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for line in reader:
            chr = line[0][3:]
            if chr in CHROMS:
                # note: uses bankers rounding
                start = round(int(line[1]) / resolution)
                end = round(int(line[2]) / resolution)
                for j in range(start, end):
                    x[chr][j] = 1

    # export data
    for i_c in CHROMS:
        np.save(osp.join(dir, i_c + '.npy'), x[i_c].astype(np.int8))

    return x


def main():
    args = getArgs()

    files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith(args.file_type)]
    names = get_names(args.dir, files)
    print(names, len(names))
    for file in files:
        aggregate_peaks(file, args.res)

    make_chromHMM_table(names, files, args)


if __name__ == '__main__':
    main()
