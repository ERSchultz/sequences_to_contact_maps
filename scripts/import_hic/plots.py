import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix


def plot_p_s_chrom(cell_line, chrom):
    dir = f'/home/erschultz/dataset_{cell_line}'
    y_combined = None
    for chrom_rep in os.listdir(dir):
        if 'rep' not in chrom_rep:
            continue
        rep = chrom_rep[-1]
        y = np.load(osp.join(dir, f'{chrom_rep}/chr{chrom}/y.npy'))
        if y_combined is None:
            y_combined = y
        else:
            y_combined += y
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-{rep}')

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_combined, 'prob')
    plt.plot(meanDist, label = f'chr{chrom}-combined', c='k')
    plot_matrix(y_combined, osp.join(dir, f'y_chr{chrom}_combined.png'), vmax='mean')

    norm_file = osp.join(dir, f'chr{chrom}_multiHiCcompare.txt')
    if osp.exists(norm_file):
        y = np.loadtxt(norm_file)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-norm', c='k', ls=':')
        plot_matrix(y, osp.join(dir, f'y_chr{chrom}_norm.png'), vmax=np.mean(y_combined))


    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Beads', fontsize=16)
    plt.legend()
    plt.savefig(osp.join(dir, f'p_s_chrom{chrom}.png'))
    plt.close()

def plot_p_s_chrom_norm(cell_line, chrom):
    '''Plots normalized replicates vs combined normalized replicate.'''
    dir = f'/home/erschultz/dataset_{cell_line}'
    for chrom_rep in os.listdir(dir):
        if 'rep' not in chrom_rep:
            continue
        rep = chrom_rep[-1]
        y = np.loadtxt(osp.join(dir, f'{chrom_rep}/chr{chrom}/y_multiHiCcompare.txt'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-{rep}')

    norm_file = osp.join(dir, f'chr{chrom}_multiHiCcompare.txt')
    if osp.exists(norm_file):
        y = np.loadtxt(norm_file)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-norm', c='k', ls=':')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Beads', fontsize=16)
    plt.legend()
    plt.savefig(osp.join(dir, f'p_s_chrom{chrom}_norm.png'))
    plt.close()


def plot_p_s_chroms(cell_line):
    """Combine (sum) replicates to create a single p(s) per chrom"""
    dir = f'/home/erschultz/dataset_{cell_line}'
    for chrom in range(1, 23):
        y = None
        for chrom_rep in os.listdir(dir):
            if 'rep' not in chrom_rep:
                continue
            y_file = osp.join(dir, f'{chrom_rep}/chr{chrom}/y.npy')
            if not osp.exists(y_file):
                continue
            y_rep = np.load(y_file)
            if y is None:
                y = y_rep
            else:
                y += y_rep
        if y is not None:
            y /= np.mean(np.diagonal(y))
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            plt.plot(meanDist, label = f'chr{chrom}')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(10**-4, None)
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Beads', fontsize=16)
    plt.legend(loc='right')
    plt.savefig(osp.join(dir, f'p_s_chroms.png'))
    plt.close()

def compare_inpt_out_Lyu():
    dir = '/home/erschultz/NormCompare/data'
    f1 = '/home/erschultz/NormCompare/data/input/GM12878/chr1/1M/001_chr1_1Mb.txt'
    f2 = '/home/erschultz/NormCompare/data/input/GM12878/chr1/1M/002_chr1_1Mb.txt'
    f3 = '/home/erschultz/NormCompare/data/output/GM12878/chr1/1M/001_chr1_1Mb-multiHiCcompare.txt'
    f4 = '/home/erschultz/NormCompare/data/output/GM12878/chr1/1M/002_chr1_1Mb-multiHiCcompare.txt'

    names = ['inp1', 'inp2', 'out1', 'out2']
    y_list = []
    m=250
    res=1000000
    for f, name in zip([f1, f2, f3, f4], names):
        y = np.loadtxt(f)
        print(y.shape)
        if 'inp' in name:
            data = y[:, -1]
            rows = y[:, -2]/res
            cols = y[:, -3]/res
            y = ss.coo_array((data, (rows, cols)), shape=(m, m)).toarray()
            print(y)
            print(np.tril(y, -1))
            y += np.tril(y, -1).T
            print(y.shape)
        # plot_matrix(y, osp.join(dir, f'{name}.png'), vmax=np.mean(y))
        y_list.append(y)

    for y, name in zip(y_list, names):
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = name)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Beads', fontsize=16)
    plt.legend()
    plt.savefig(osp.join(dir, f'p_s.png'))
    plt.close()



if __name__ == '__main__':
    # compare_inpt_out_Lyu()
    plot_p_s_chrom('11_17_23', 17)
    plot_p_s_chrom_norm('11_17_23', 17)
    plot_p_s_chroms('11_17_23')
