import csv
import json
import multiprocessing
import os
import os.path as osp
import pickle
import subprocess
import time

import bioframe  # open2c https://github.com/open2c/bioframe
import cooltools  # https://cooltools.readthedocs.io/en/latest/index.html
import hicrep
import matplotlib.pyplot as plt
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

def pre_to_hic(dir, jobs = 19):
    # create hic file with Pre
    t0 = time.time()

    sc_files = os.listdir(osp.join(dir, 'samples'))
    jar_file = '/home/erschultz/juicer/scripts/common/juicer_tools.jar'
    resolutions = '2500000,1000000,500000,250000,100000,50000,25000,10000,1000'

    mapping = []
    for i, sc_file in enumerate(sc_files):
        if sc_file in {'1CDS2.346', '1CDS2.513'}:
            sc_dir = osp.join(dir, 'samples', sc_file)
            ifile = osp.join(sc_dir, 'juicer_pre_ifile.txt')
            ofile = osp.join(sc_dir, f'adj.hic')
            command = f'java -Xmx2g -jar {jar_file} pre {ifile} {ofile} mm9 -d -r {resolutions}'
            log_file = osp.join(sc_dir, 'pre.log')
            mapping.append((command, log_file))

    print(mapping)

    with multiprocessing.Pool(jobs) as p:
        p.starmap(bash, mapping)

    tf = time.time()
    print(f'Finished writing hic files in {np.round(tf-t0, 1)} seconds')

def hic_to_cool(dir, resolution = None):
    sc_files = sorted(os.listdir(osp.join(dir, 'samples')))
    errors = []
    for i, sc_file in enumerate(sc_files):
        sc_dir = osp.join(dir, 'samples', sc_file)
        ifile = osp.join(sc_dir, f'adj.hic')
        try:
            if resolution is not None:
                ofile = osp.join(sc_dir, f'adj_{resolution}.cool')
                hic2cool_convert(ifile, ofile, resolution)
            else:
                ofile = osp.join(sc_dir, 'adj.mcool')
                hic2cool_convert(ifile, ofile)
        except Exception as e:
            errors.append((ifile, e))

    for ifile, e in errors:
        print(ifile)
        print(e)
        print()


def cell_cycle_phasing(dir):
    samples = [osp.join(dir, 'samples', f) for f in os.listdir(osp.join(dir, 'samples'))]

    # Use bioframe to fetch the genomic features from the UCSC.
    chromsizes = bioframe.fetch_chromsizes('mm9')
    cens = bioframe.fetch_centromeres('mm9')
    # create a view with chromosome arms using chromosome sizes and definition of centromeres
    arms = bioframe.make_chromarms(chromsizes,  cens)

    # convert 'chr{i}' to '{i}'
    arms['chrom'] = arms['chrom'].str.replace('chr','')
    arms = arms[arms['chrom'] != 'M'] # ignore mitochondrial
    arms = arms[arms['chrom'] == '10'] # TODO only considering chr10

    mapping = [(f, arms) for f in samples]
    # with multiprocessing.Pool(15) as p:
    #     p.starmap(contact_distance_profile, mapping)

    phase_dict = {}
    for f in samples:
        ifile = osp.join(f, 'p_s.npy')
        p_s = np.load(ifile)
        # TODO, check for off by one error
        near = np.sum(p_s[2:220])
        mitotic = np.sum(p_s[220:1130])
        tot = np.nansum(p_s)
        prcnt_near = near / tot * 100
        prcnt_mitotic = mitotic / tot * 100

        # assign initial phase
        if prcnt_mitotic < 30 and prcnt_near < 50:
            phase = 'post-M'
        elif prcnt_near > 50 and (prcnt_near + 1.8 * prcnt_mitotic) > 100:
            phase = 'pre-M'
        elif prcnt_near < 63:
            phase = 'G1'
        elif prcnt_near < 78.5:
            phase = 'S'
        else:
            phase = 'G2'

        phase_dict[f] = phase

        with open(osp.join(f, 'phase.txt'), 'w') as f:
            f.write(phase)

    with open(osp.join(dir, 'phase_dict.json'), 'w') as f:
        json.dump(phase_dict, f, indent = 2)

def read_count(dir):
    samples = [osp.join(dir, 'samples', f) for f in os.listdir(osp.join(dir, 'samples'))]
    samples = [f for f in samples if osp.isdir(f)]

    # read_count_dict = {}
    # for f in samples:
    #     clr, _ = hicrep.utils.readMcool(osp.join(f, 'adj.mcool'), 2500000)
    #     y_list = []
    #     for chrom in clr.chromnames:
    #         y_list.append(clr.matrix(balance=False).fetch(f'{chrom}'))
    #     read_count = 0
    #     for y in y_list:
    #         read_count += np.sum(np.triu(y))
    #     read_count_dict[f] = int(read_count)
    #
    #     with open(osp.join(f, 'read_count.txt'), 'w') as f:
    #         f.write(str(read_count))
    #
    # with open(osp.join(dir, 'samples/read_count_dict.json'), 'w') as f:
    #     json.dump(read_count_dict, f, indent = 2)


    with open(osp.join(dir, 'samples/read_count_dict.json'), 'r') as f:
        read_count_dict = json.load(f)

    values = list(read_count_dict.values())
    min_val = np.min(values)
    max_val = np.max(values)
    plt.hist(values, bins = np.logspace(np.log10(min_val), np.log10(max_val), 20))
    plt.xscale('log')
    plt.show()

def contact_distance_profile(sample, arms):
    ifile = osp.join(sample, 'adj.mcool')
    resolution = 10000
    clr, _ = hicrep.utils.readMcool(ifile, resolution)

    # select only those chromosomes available in cooler
    arms = arms[arms.chrom.isin(clr.chromnames)].reset_index(drop=True)

    # calculate P(s)
    cvd = cooltools.expected_cis(clr=clr,
                                view_df=arms,
                                smooth=False,
                                aggregate_smoothed=False,
                                clr_weight_name=None,
                                nproc=1)

    cvd['s_bp'] = cvd['dist']* resolution

    # savetxt
    for region in arms['name']:
        p_s = cvd['count.avg'].loc[cvd['region1']==region]
        p_s = np.array(p_s)
        np.save(osp.join(sample, 'p_s.npy'), p_s)
        return

    # plot
    f, ax = plt.subplots(1,1)

    for region in arms['name']:
        ax.loglog(
            cvd['s_bp'].loc[cvd['region1']==region],
            cvd['count.avg'].loc[cvd['region1']==region], label = region
        )
    ax.set_xlabel('separation, bp0', fontsize = 16)
    ax.set_ylabel('IC contact frequency', fontsize = 16)
    ax.set_aspect(1.0)
    ax.grid(lw=0.5)
    ax.legend()
    ax.axline((22600, 1), (22600, 2), color = 'k')
    ax.axline((2200000, 1), (2200000, 2), color = 'k')
    ax.axline((22600000, 1), (22600000, 2), color = 'k')
    plt.tight_layout()
    plt.savefig(osp.join(sample, 'P_s.png'))

def main():
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017'
    # adj_to_pre(dir)
    # pre_to_hic(dir)
    hic_to_cool(dir, 500000)
    # cell_cycle_phasing(dir)
    # read_count(dir)
    # timer(dir)

def timer(dir):
    # track progress of pre_to_hic
    sc_files = os.listdir(osp.join(dir, 'samples'))
    sc_files = [f for f in sc_files if not f.endswith('.json')]

    mapping = []
    done = 0
    undone = 0
    for sc_file in sc_files:
        ofile = osp.join(dir, 'samples', sc_file, f'adj.hic')
        t = osp.getmtime(ofile)
        # print(ofile, t)
        if t < 1661500000:
            undone += 1
        else:
            done += 1
    print(done / (undone + done))

if __name__ == '__main__':
    main()
