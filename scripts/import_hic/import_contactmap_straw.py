'''
Functions for importing experimental Hi-C contact maps.

import_contactmap_straw interacts with .hic files.
Most other functions are some form of wrapper.
'''

import csv
import multiprocessing
import os
import os.path as osp

import bioframe  # https://github.com/open2c/bioframe
import hicstraw  # https://github.com/aidenlab/straw
import numpy as np
import pandas as pd
from pylib.utils.hic_utils import rescale_p_s_1
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log
from utils import *


def import_contactmap_straw(odir, hic_filename, chrom, start,
                            end, resolution, norm='NONE', multiHiCcompare=False):
    '''
    Load .hic file with hicstraw and write to disk as y.npy. Experimental details
    are logged in odir/import.log.

    Inputs:
        odir: output directory
        hic_filname: path to .hic file
        chrom: chromosome
        start: start basepair
        end: end basepair
        resolution: Hi-C resolution in basepairs
        norm: Hi-C normlalization method
        multiHiCcompare: True to write contact map in sparse format used by
                            multiHiCcompare R package
    '''
    basepairs = f"{chrom}:{start}:{end}"
    print(basepairs, odir)
    result = hicstraw.straw("observed", norm, hic_filename, basepairs, basepairs, "BP", resolution)
    hic = hicstraw.HiCFile(hic_filename)

    m = int((end - start) / resolution)
    y_arr = np.zeros((m, m))
    output = []
    for row in result:
        i = int((row.binX - start) / resolution)
        j = int((row.binY - start) / resolution)
        if i >= m or j >= m:
            continue
        try:
            y_arr[i, j] = row.counts
            y_arr[j, i] = row.counts
            if multiHiCcompare:
                output.append([chrom, row.binX, row.binY, row.counts])
        except Exception as e:
            print(e)
            print(row.binX, row.binY, row.counts, i, j)

    if np.max(y_arr) == 0:
        print(f'{odir} had no reads')
        return

    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    if multiHiCcompare:
        with open(osp.join(odir, 'y_sparse.txt'), 'w') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerows(output)

    m, _ = y_arr.shape

    with open(osp.join(odir, 'import.log'), 'w') as f:
        if isinstance(chrom, str):
            chrom = chrom.strip('chr')
        f.write(f'{hic_filename}\nchrom={chrom}\nstart={start}\nend={end}\n')
        f.write(f'resolution={resolution}\nbeads={m}\nnorm={norm}\n')
        f.write(f'genome={hic.getGenomeID()}')

    np.save(osp.join(odir, 'y.npy'), y_arr)
    print(f'{odir} done')

def import_wrapper(odir, filename_list, resolution, norm, m,
                    i, ref_genome, chroms, seed):
    '''
    Wrapper function for import_contactmap_straw. Iterates through chromosomes
    of input .hic files and downloads non-overlapping contact maps of size m.

    Inputs:
        odir: output directory
        filename_list: list of .hic files to download
        resolution: Hi-C resolution in basepairs
        norm: Hi-C normalization method
        m: number of rows/cols in contact maps that will be downloaded
        i: start index for output folders
        ref_genome: reference genome build (used to restrict to regions with good coverage)
        chroms: list of chromosomes to download
        seed: random seed (non-overlapping Hi-C maps are downloaded with a random gap in between)
    '''
    rng = np.random.default_rng(seed)
    if isinstance(filename_list, str):
        filename_list = [filename_list]
    chromsizes = bioframe.fetch_chromsizes(ref_genome)
    mapping = []
    for filename in filename_list:
        for chromosome in chroms:
            start_mb = 0
            start = start_mb * 1000000
            end = start + resolution * m
            end_mb = end / 1000000
            while end < chromsizes[f'chr{chromosome}']:
                for region in HG19_BAD_REGIONS[chromosome].split(','):
                    region = region.split('-')
                    region = [int(d) for d in region]
                    if intersect((start_mb, end_mb), region):
                        start_mb = region[1] # skip to end of bad region
                        start_mb += rng.choice(np.arange(6)) # add random shift
                        break
                else:
                    print(f'i={i}: chr{chromosome} {start_mb}-{end_mb}')
                    sample_folder = osp.join(odir, f'sample{i}')
                    mapping.append((sample_folder, filename, chromosome, start,
                                    end, resolution, norm))
                    i += 1
                    start_mb = end_mb

                start = int(start_mb * 1000000)
                end = start + resolution * m
                end_mb = end / 1000000

    with multiprocessing.Pool(15) as p:
        p.starmap(import_contactmap_straw, mapping)

def single_experiment_dataset(filename, dataset, resolution, m,
                                norm='NONE', start_index=1, ref_genome='hg19',
                                chroms=range(1,23), seed=None):
    dir = '/home/erschultz'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    odir = osp.join(data_folder, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    import_wrapper(odir, filename, resolution, norm, m, start_index, ref_genome, chroms, seed)

def entire_chromosomes(filename, dataset, resolution,
                        norm='NONE', ref_genome='hg19',
                        chroms=range(1,23), odir=None, multiHiCcompare=False):
    dir = '/home/erschultz'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if odir is None:
        odir = osp.join(data_folder, f'chroms_{resolution//1000}k')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    chromsizes = bioframe.fetch_chromsizes(ref_genome)
    mapping = []
    for i, chromosome in enumerate(chroms):
        i += 1 # switch to 1-based indexing
        start = 0
        end = chromsizes[f'chr{chromosome}']
        print(f'i={i}: chr{chromosome} {start}-{end}')
        sample_folder = osp.join(odir, f'chr{chromosome}')
        mapping.append((sample_folder, filename, chromosome, start,
                        end, resolution, norm, multiHiCcompare))

    with multiprocessing.Pool(15) as p:
        p.starmap(import_contactmap_straw, mapping)

    for chr in chroms:
        y = np.load(osp.join(odir, f'chr{chr}', 'y.npy'))
        plot_matrix(y, osp.join(odir, f'chr{chr}', 'y.png'), vmax='mean')


def entire_chromosomes_list(filenames, dataset, resolution=50000,
                        ref_genome='hg19', chroms=range(1,23)):
    for i, filename in enumerate(filenames):
        odir = osp.join('/home/erschultz', dataset, f'chroms_rep{i}')
        entire_chromosomes(filename, dataset, resolution, 'NONE', ref_genome,
                            chroms, odir, True)

def split(in_dataset, out_dataset, m, chroms=range(1,23), start_index=1,
                        resolution=50000, ref_genome='hg19', seed=None,
                        file = 'y_multiHiCcompare.txt', scale=None):
    data_dir = osp.join('/home/erschultz', in_dataset)
    out_data_dir = osp.join('/home/erschultz', out_dataset)
    if not osp.exists(out_data_dir):
        os.mkdir(out_data_dir, mode=0o755)
    samples_dir = osp.join(out_data_dir, 'samples')
    if not osp.exists(samples_dir):
        os.mkdir(samples_dir, mode=0o755)

    cell_lines = ['50k']

    i = start_index
    chromsizes = bioframe.fetch_chromsizes(ref_genome)
    for cell_line in cell_lines:
        print(cell_line)
        for chrom in chroms:
            chrom_dir = osp.join(data_dir, f'chroms_{cell_line}/chr{chrom}')
            y_file = osp.join(chrom_dir, file)
            if y_file.endswith('.txt'):
                y = np.loadtxt(y_file)
            elif y_file.endswith('.npy'):
                y = np.load(y_file)
            size = len(y)
            diag = y.diagonal() == 0
            ind = np.arange(0, len(diag))

            import_log = load_import_log(chrom_dir)

            start = 0
            end = start + m
            while end < size:
                if np.sum(diag[start:end]) > 0:
                    start = end - np.argmax(np.flip(diag[start:end]))
                    end = start + m
                    continue

                print(f'\ti={i}: chr{chrom} {start}-{end}')
                odir = osp.join(samples_dir, f'sample{i}')
                if not osp.exists(odir):
                    os.mkdir(odir, mode=0o755)

                with open(osp.join(odir, 'import.log'), 'w') as f:
                    if isinstance(chrom, str):
                        chrom = chrom.strip('chr')
                    f.write(f'{import_log["url"]}\nchrom={chrom}\n')
                    f.write(f'resolution={resolution}\nbeads={m}\nnorm={import_log["norm"]}\n')
                    f.write(f'start={int(start*resolution)}\nend={int(end*resolution)}\n')
                    f.write(f'genome={import_log["genome"]}')


                y_i = y[start:end,start:end]
                if scale is not None:
                    y_i = rescale_p_s_1(y_i, scale)
                    np.fill_diagonal(y_i, 1)
                np.save(osp.join(odir, 'y.npy'), y_i)
                plot_matrix(y_i, osp.join(odir, 'y.png'), vmax='mean')
                i += 1
                start = end
                end = start + m


def Su2020imr90():
    sample_folder = '/home/erschultz/Su2020/samples/sample4'
    filename='https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic'
    filename='/home/erschultz/Su2020/ENCFF281ILS.hic'

    resolution = 10000
    start = 14000001
    end = start + 512*5*resolution
    import_contactmap_straw(sample_folder, filename, 'chr21', start, end, resolution, 'NONE')

def make_latex_table():
    files = ALL_FILES_NO_GM12878.copy()
    files.extend(VALIDATION_FILES)
    cell_lines = []
    for f in files:
        f_split = f.split(os.sep)
        cell_line = f_split[-3]
        if cell_line == 'GSE104333':
            cell_line = 'HCT116'
        elif cell_line == 'ENCFF177TYX':
            cell_line = 'HL-60'
        else:
            cell_line = cell_line.upper()
        cell_lines.append(cell_line)

    use_case = ['Training']*len(ALL_FILES_NO_GM12878)
    use_case.extend(['Validation']*len(VALIDATION_FILES))

    print(len(cell_lines), len(files), len(use_case))

    d = {'Cell Line':cell_lines, "Use": use_case, "File": files, }
    df = pd.DataFrame(data = d)
    pd.set_option('display.max_colwidth', -1)
    print(df)
    print(df.to_latex(index = False))
    df.to_csv('/home/erschultz/TICG-chromatin/figures/tableS1.csv', index = False)


if __name__ == '__main__':
    # make_latex_table()
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
    #                             'dataset_interp_test', 10000, 512*5, chroms=[1])
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
    #                             'dataset_02_04_23', 10000, 512*5)
    # entire_chromosomes("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
    #                     'dataset_02_05_23', 50000)
    # Su2020imr90()
    # entire_chromosomes_list(GM12878_REPLICATES, 'dataset_gm12878', resolution=5000, chroms=[1,2])
    # entire_chromosomes('https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic',
                        # 'dataset_gm12878_25k', resolution=25000, chroms=[1,2])
    # entire_chromosomes_list(ALL_FILES_in_situ, 'dataset_11_20_23')
    # entire_chromosomes(HCT116_RAD21KO, 'dataset_HCT116_RAD21_KO', 50000)
    split('dataset_HCT116_RAD21_KO', 'dataset_HCT116_RAD21_KO', 1024, file='y.npy', scale=1e-1)
    # split('dataset_11_20_23', 'dataset_12_01_23', 512, file='y_multiHiCcompare.txt')
    # split('dataset_11_20_23', 'dataset_12_06_23', 512, file='y.npy', scale=1e-1)
    # split('dataset_gm12878_25k', 'dataset_gm12878_25k', 1024, file='y.npy', scale=1e-1, resolution=25000)
    # entire_chromosomes('https://hicfiles.s3.amazonaws.com/external/dixon/mm9-hindiii/split-read-run.hic',
                        # 'dataset_mm9', resolution=100000, chroms=[1,2], ref_genome='mm9')
