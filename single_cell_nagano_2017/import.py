import csv
import multiprocessing
import os
import os.path as osp
import pickle
import subprocess
import time

import numpy as np
from hic2cool import hic2cool_convert  # https://github.com/4dn-dcic/hic2cool


def bash(command, log_file):
    process = subprocess.Popen(command.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()

    with open(log_file, 'wb') as f:
        f.write(output)
        if error is not None:
            f.write(error)
            print(f'Error with {command}')


def chr_to_int(chr, max_somatic = 22):
    if chr.isdigit():
        return int(chr)
    elif chr == 'X':
        return max_somatic + 1
    elif chr == 'Y':
        return max_somatic + 2
    else:
        return None

def adj_to_pre(dir):
    # load GATC.fends
    GATC_dict_file = osp.join(dir, 'GATC_dict.pickle')
    if osp.exists(GATC_dict_file):
        with open(GATC_dict_file, 'rb') as f:
            GATC_dict = pickle.load(f)
    else:
        GATC_dict = {}
        with open(osp.join(dir, 'GATC.fends'), 'r') as f:
            f.readline()
            reader = csv.reader(f, delimiter = '\t')
            for line in reader:
                fend, chr, coord = line
                GATC_dict[fend] = (chr, coord)
        with open(GATC_dict_file, 'wb') as f:
            pickle.dump(GATC_dict, f)

    sc_files = os.listdir(osp.join(dir, 'samples'))
    print(f'{len(sc_files)} single-cell hic maps')
    for i, sc_file in enumerate(sc_files):
        if i % 10 == 0:
            print(i, sc_file)

        sc_dir = osp.join(dir, 'samples', sc_file)
        ofile = osp.join(sc_dir, 'juicer_pre_ifile.txt')
        rows = []
        with open(osp.join(sc_dir, 'adj'), 'r') as f:
            f.readline()
            reader = csv.reader(f, delimiter = '\t')
            for line in reader:
                strand = 0
                fend1, fend2, count = line
                chr1, coord1 = GATC_dict[fend1]
                chr2, coord2 = GATC_dict[fend2]
                if chr1 < chr2:
                    row = [strand, chr1, coord1, 0, strand, chr2, coord2, 1]
                else:
                    row = [strand, chr2, coord2, 0, strand, chr1, coord1, 1]
                rows.append(row)

        # sort rows
        rows = sorted(rows, key = lambda x: (chr_to_int(x[1]), chr_to_int(x[5])))

        with open(ofile, 'w', newline='') as f:
            wr = csv.writer(f, delimiter ='\t')
            wr.writerows(rows)
    print('Finished writing juicer_pre_ifiles')

def pre_to_hic(dir, jobs = 15):
    # create hic file with Pre
    t0 = time.time()

    sc_files = os.listdir(osp.join(dir, 'schic_hyb_1CDS2_adj_files'))
    jar_file = '/home/erschultz/juicer/scripts/common/juicer_tools.jar'
    resolutions = '2500000,1000000,500000,250000,100000,50000,25000,10000'

    mapping = []
    for i, sc_file in enumerate(sc_files):
        sc_dir = osp.join(dir, 'samples', sc_file)
        ifile = osp.join(sc_dir, 'juicer_pre_ifile.txt')
        ofile = osp.join(sc_dir, f'adj.hic')
        command = f'java -Xmx2g -jar {jar_file} pre {ifile} {ofile} mm9 -d -r {resolutions}'
        log_file = osp.join(sc_dir, 'pre.log')
        mapping.append((command, log_file))

    with multiprocessing.Pool(jobs) as p:
        p.starmap(bash, mapping)

    tf = time.time()
    print(f'Finished writing hic files in {np.round(tf-t0, 1)} seconds')

def hic_to_cool(dir):
    sc_files = os.listdir(osp.join(dir, 'samples'))
    mapping = []
    for i, sc_file in enumerate(sc_files):
        sc_dir = osp.join(dir, 'samples', sc_file)
        print(sc_file)
        ifile = osp.join(sc_dir, f'adj.hic')
        ofile = osp.join(sc_dir, f'adj.mcool')
        hic2cool_convert(ifile, ofile)

def main():
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017'
    # adj_to_pre(dir)
    # pre_to_hic(dir)
    hic_to_cool(dir)

if __name__ == '__main__':
    main()
