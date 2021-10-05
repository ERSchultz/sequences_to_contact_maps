import os
import os.path as osp

import pandas as pd
import csv
import argparse

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

def make_chromHMM_table(names, files, args):
    ofile = osp.join(args.dir, 'cell_mark_file_table.tsv')
    with open(ofile, 'w', newline = '') as f:
        wr = csv.writer(f, delimiter = '\t')
        for name, file in zip(names, files):
            file = osp.split(file)[1]
            wr.writerow([args.cell_line, name, file])


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=osp.join('chip_seq_data','aligned_reads'), help='directory of chip-seq data')
    parser.add_argument('--file_type', default='.bam', help='file format')
    parser.add_argument('--cell_line', default='HTC116', help='cell line')
    args = parser.parse_args()

    return args


def main():
    args = getArgs()

    files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith(args.file_type)]
    names = get_names(args.dir, files)
    print(names, len(names))

    make_chromHMM_table(names, files, args)


if __name__ == '__main__':
    main()
