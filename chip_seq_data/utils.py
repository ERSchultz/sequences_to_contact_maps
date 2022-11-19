import argparse
import csv
import os
import os.path as osp

import numpy as np
import pandas as pd

#Autologous chroms
CHROMS = [str(ele) for ele in range(1,23)]
#Sex chrom: X only
CHROMS.append("X")

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
    ofile = osp.join(args.dir, 'marks.txt')
    print(ofile)
    with open(ofile, 'w', newline = '') as f:
        wr = csv.writer(f, delimiter = '\t')
        for name, file in zip(names, files):
            file = osp.split(file)[1]
            wr.writerow([args.cell_line, name, file])

def save_chip_for_CHROMHMM(chips, names, args):
	for i, chrom in enumerate(chips):
		ofile = osp.join(args.dir, 'processed', f'chr{CHROMS[i]}_binary.txt')
		combined_marks = np.zeros((len(chrom[0]), len(chrom)))
		for j, mark in enumerate(chrom):
			combined_marks[:, j] = mark[:, 1]
			with open(ofile, 'w', newline = '') as f:
				wr = csv.writer(f, delimiter = '\t')
				wr.writerow([args.cell_line, f"chr{CHROMS[i]}"])
				wr.writerow(names)
				wr.writerows(combined_marks.astype(np.int8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=osp.join('chip_seq_data', 'HCT116', 'aligned_reads'),
                        help='directory of chip-seq data')
    parser.add_argument('--file_type', default='.bam', help='file format')
    parser.add_argument('--cell_line', default='HCT116', help='cell line')
    args = parser.parse_args()

    files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith(args.file_type)]
    names = get_names(args.dir, files)
    print(names, len(names))

    make_chromHMM_table(names, files, args)


if __name__ == '__main__':
    main()
