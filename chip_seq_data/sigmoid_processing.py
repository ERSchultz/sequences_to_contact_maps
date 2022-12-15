import argparse
import csv
import decimal
import multiprocessing
import os
import os.path as osp

import bioframe
import numpy as np
import pyBigWig
from bw_to_npy import bw_to_npy
from chipseqPipeline import ChipseqPipeline, Normalize, Sigmoid, Smooth
from utils import CHROMS, get_names, make_chromHMM_table


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', default='HCT116', help='cell line')
    parser.add_argument('--genome_build', default='hg19')
    parser.add_argument('--res', default=50000, help='resolution for chip-seq data')
    parser.add_argument('--file_type', default='.bigWig', help='file type')
    parser.add_argument('--data_format', default='fold_change_control', help='data format')

    args = parser.parse_args()

    args.dir = osp.join('chip_seq_data', args.cell_line, args.genome_build, args.data_format)
    args.data_dir = osp.join(args.dir, 'bigWigFiles')

    args.odir = osp.join(args.dir, 'processed_sigmoid')
    if not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)

    return args

def main():
    '''Use Soren's strategy to process chip-seq'''
    args = getArgs()

    files = [osp.join(args.data_dir, file) for file in os.listdir(args.data_dir) if file.endswith(args.file_type)]
    names = get_names(args.dir, files)
    print(names, len(names))

    # mapping = zip(files, [args]*len(files))
    # args.mode = 'max'
    # args.nucl = False
    # with multiprocessing.Pool(12) as p:
    #     p.starmap(bw_to_npy, mapping)

    chipseq_pipeline = ChipseqPipeline([Smooth(), Normalize(), Sigmoid()], False)
    for file, name in zip(files, names):
        folder = osp.split(file)[1].split('.')[0]
        print(folder)
        for chr in CHROMS:
            print(chr)
            arr = np.load(osp.join(args.dir, folder, f'{chr}.npy'), allow_pickle = True)
            arr = arr[:, 1].astype(np.float64)
            seq = chipseq_pipeline.fit(arr)
            np.save(osp.join(args.odir, f'chr{chr}_{name}.npy'), seq)



if __name__ == '__main__':
    main()
