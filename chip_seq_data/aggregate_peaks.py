import argparse
import csv
import os
import os.path as osp

import bioframe
import numpy as np
from utils import CHROMS, get_names, make_chromHMM_table


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', default='HCT116', help='cell line')
    parser.add_argument('--genome_build', default='hg19')
    parser.add_argument('--res', default=50000, help='resolution for chip-seq data')
    parser.add_argument('--file_type', default='.bed', help='file format')

    args = parser.parse_args()

    args.dir = osp.join('chip_seq_data', args.cell_line, args.genome_build, 'narrow_peaks')

    return args

def aggregate_peaks(ifile, name, args):
    '''
    Use approach from Zhou et al (DeepSEA) to aggregate peaks from Chip-seq data.
    Iterate though bins and if bin overlaps with peak by at least 50%, assign that label

    Reference:
        https://doi.org/10.1038/nmeth.3547
    Inputs:
        ifile: file location of single chip-seq narrow peak track in bed format
        name: name of chip-seq mark
        args: argparse object
    Outputs:
        x: dictionary mapping chr (int) : aggregated peaks (numpy array)
    '''
    odir = ifile.split('.')[0]
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    x = {} # chr : np array of aggregated peaks
    chromsizes = bioframe.fetch_chromsizes(args.genome_build)
    for chr in CHROMS:
        x[chr] = np.zeros(int(chromsizes[f'chr{chr}'] / args.res))

    with open(ifile, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for line in reader:
            chr = line[0][3:]
            if chr in CHROMS:
                # note: uses bankers rounding
                start = round(int(line[1]) / args.res) # start of called peak
                end = round(int(line[2]) / args.res) # end of peak
                for j in range(start, end):
                    x[chr][j] = 1

    # export data
    for chr in CHROMS:
        np.save(osp.join(odir, f'{args.cell_line}_{name}_chr{chr}.npy'), x[chr].astype(np.int8))

    return x


def main():
    args = getArgs()

    files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith(args.file_type)]
    names = get_names(args.dir, files)
    print(names, len(names))
    for file, name in zip(files, names):
        aggregate_peaks(file, name, args)

    make_chromHMM_table(names, files, args)


if __name__ == '__main__':
    main()
