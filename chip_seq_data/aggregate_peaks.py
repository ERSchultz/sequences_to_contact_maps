import numpy as np
import argparse
import csv
import os.path as osp
import os
import pandas as pd
import decimal

#Autologous chroms
CHROMS = [str(ele) for ele in range(1,23)]
#Sex chrom: X only
CHROMS.append("X")

CHROM_LENGTHS = {'1': 248956422,
                '2': 242193529,
                '3': 198295559,
                '4': 198295559,
                '5': 181538259,
                '6': 170805979,
                '7': 159345973,
                '8': 145138636,
                '9': 138394717,
                '10': 133797422,
                '11': 135086622,
                '12': 133275309,
                '13': 114364328,
                '14': 107043718,
                '15': 101991189,
                '16': 90338345,
                '17': 83257441,
                '18': 80373285,
                '19': 58617616,
                '20': 64444167,
                '21': 46709983,
                '22': 50818468,
                'X': 156040895}


def aggregate_peaks(ifile, resolution):
    '''
    Uses approach from Zhou et al (DeepSEA) to aggregate peaks from Chip-seq data.

    Reference:
        https://doi.org/10.1038/nmeth.3547

    Inputs:
        ifile: file location of single chip-seq narrow peak track in bed format
        ofile: file location to save resulting aggregation
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


def get_names(dir, files):
    metadata_file = osp.join(dir, 'metadata.tsv')
    if osp.exists(metadata_file):
        metadata = pd.read_csv(metadata_file, sep = '\t')
    else:
        raise Exception('metadata.tsv does not exist')

    names = []
    for file in files:
        accession = osp.split(file)[-1].split('.')[0]
        target = metadata[metadata['File accession'] == accession]['Experiment target'].item()
        name = target.split('-')[0]
        names.append(name)

    names_set = set(names)
    if len(names) != len(names_set):
        print('Warning: duplicate names')

    return names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=osp.join('chip_seq_data','bedFiles'), help='directory of chip-seq data')
    parser.add_argument('--res', default = 200, help='resolution for chip-seq data')
    args = parser.parse_args()


    files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('bed')]
    names = get_names(args.dir, files)
    print(names, len(names))
    for file in files:
        aggregate_peaks(file, args.res)


if __name__ == '__main__':
    main()
