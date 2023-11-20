import csv
import multiprocessing
import os
import os.path as osp

import bioframe  # https://github.com/open2c/bioframe
import hicstraw  # https://github.com/aidenlab/straw
import numpy as np
import pandas as pd
from utils import *


def import_contactmap_straw(sample_folder, hic_filename, chrom, start,
                            end, resolution, norm='NONE', multiHiCcompare=False):
    basepairs = f"{chrom}:{start}:{end}"
    print(basepairs, sample_folder)
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
        print(f'{sample_folder} had no reads')
        return

    if not osp.exists(sample_folder):
        os.mkdir(sample_folder, mode = 0o755)

    if multiHiCcompare:
        with open(osp.join(sample_folder, 'y_sparse.txt'), 'w') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerows(output)

    m, _ = y_arr.shape

    with open(osp.join(sample_folder, 'import.log'), 'w') as f:
        if isinstance(chrom, str):
            chrom = chrom.strip('chr')
        f.write(f'{hic_filename}\nchrom={chrom}\nstart={start}\nend={end}\n')
        f.write(f'resolution={resolution}\nbeads={m}\nnorm={norm}\n')
        f.write(f'genome={hic.getGenomeID()}')

    np.save(osp.join(sample_folder, 'y.npy'), y_arr)
    print(f'{sample_folder} done')

def import_wrapper(odir, filename_list, resolution, norm, m,
                    i, ref_genome, chroms, seed):
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
                                norm='NONE', i=1, ref_genome='hg19',
                                chroms=range(1,23), seed=None):
    dir = '/home/erschultz'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    odir = osp.join(data_folder, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    import_wrapper(odir, filename, resolution, norm, m, i, ref_genome, chroms, seed)

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


def multiHiCcompare_files(filenames, dataset, resolution=50000, ref_genome='hg19',
                        chroms=range(1,23)):
    for i, filename in enumerate(filenames):
        odir = osp.join('/home/erschultz', dataset, f'chroms_rep{i}')
        entire_chromosomes(filename, dataset, resolution, 'NONE', ref_genome,
                            chroms, odir, True)


def mixed_experimental_dataset(dataset, resolution, m, norm='NONE',
                                i=1, ref_genome='hg19',
                                chroms=range(1,23), files=ALL_FILES,
                                seed=None):
    dir = '/home/erschultz'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    odir = osp.join(data_folder, 'samples')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    import_wrapper(odir, files, resolution, norm, m, i , ref_genome, chroms, seed)

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
                                # 'dataset_02_04_23', 10000, 512*5)
    # entire_chromosomes("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
    #                     'dataset_02_05_23', 50000)
    # single_experiment_dataset("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-untreated_combined_30.hic",
    #                             'dataset_HCT116', 10000, 512*5, i=10, chroms=[2])
    # single_experiment_dataset("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-treated_6hr_combined_30.hic",
    #                             'dataset_HCT116_RAD21_KO', 10000, 512*5, chroms=[2])
    # Su2020imr90()
    # multiHiCcompare_files(GM12878_REPLICATES, 'dataset_gm12878')
    # multiHiCcompare_files(HMEC_REPLICATES, 'dataset_hmec')
    multiHiCcompare_files(ALL_FILES_in_situ, 'dataset_11_17_23', chroms=[17])
