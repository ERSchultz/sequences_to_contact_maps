import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import seaborn as sns
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log


def example_figure(dataset, sample):
    dir = osp.join(f'/home/erschultz/{dataset}')
    dir_raw = osp.join(dir, f'samples_10k/sample{sample}')
    y_raw = np.load(osp.join(dir_raw, 'y.npy'))
    # dir_interp = osp.join(dir, f'samples/sample{sample+200}')
    # y_interp = np.load(osp.join(dir_interp, 'y.npy'))
    y_interp = np.load(osp.join(dir_raw, 'Interpolation/zeros_mappability-0.7',
                                'y_interpolate_zeros_mappability-0.7.npy'))

    # get ticks in mb
    result = load_import_log(dir_raw)
    start = result['start_mb']
    end = result['end_mb']
    resoluton = result['resolutoin_mb']
    print(start, end, resolution)
    scale = 700
    all_ticks = np.arange(0, 2560, scale)
    all_ticks = np.append(all_ticks, 2560)
    print(all_ticks, len(all_ticks))

    all_tick_labels = np.arange(start, end, resolution*scale)
    all_tick_labels = np.append(all_tick_labels, end)
    print(all_tick_labels, len(all_tick_labels))
    all_tick_labels = [f'{i} Mb' for i in all_tick_labels]
    # return

    fig, (ax1, ax2, axcb) = plt.subplots(1, 3,
                                    gridspec_kw={'width_ratios':[1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2)
    vmin = 0
    vmax = np.mean(y_raw)

    s1 = sns.heatmap(y_raw, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                    ax = ax1, cbar = False)
    s1.set_title('Raw Hi-C', fontsize = 16)
    s1.set_yticks(all_ticks, labels = all_tick_labels, rotation='horizontal')
    s1.set_xticks(all_ticks, labels = all_tick_labels, rotation='horizontal')
    s2 = sns.heatmap(y_interp, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                    ax = ax2, cbar_ax = axcb)
    s2.set_title('Interpolation', fontsize = 16)
    s2.set_xticks(all_ticks, labels = all_tick_labels, rotation='horizontal')
    s2.set_yticks([])
    # axcb.yaxis.set_ticks_position('left')

    # fig.suptitle(f'Sample {sample}')
    plt.tight_layout()
    plt.savefig(osp.join(dir_raw, 'interp_figure.png'))
    plt.close()

def plot_chrom(cell_line, chrom):
    dir = f'/home/erschultz/dataset_{cell_line}'
    for chrom_rep in sorted(os.listdir(dir)):
        if 'rep' not in chrom_rep:
            continue
        rep = chrom_rep[-1]
        y = np.load(osp.join(dir, f'{chrom_rep}/chr{chrom}/y.npy'))
        read_count = np.sum(np.triu(y))
        print(rep, read_count)
        plot_matrix(y, osp.join(dir, f'{chrom_rep}/chr{chrom}/y.png'), vmax='mean', title=f'Read Count = {read_count}')

        # y = np.loadtxt(osp.join(dir, f'{chrom_rep}/chr{chrom}/y_multiHiCcompare.txt'))
        # plot_matrix(y, osp.join(dir, f'{chrom_rep}/chr{chrom}/y_norm.png'), vmax='mean')


def plot_p_s_chrom(cell_line, chrom):
    dir = f'/home/erschultz/dataset_{cell_line}'
    y_combined = None
    for chrom_rep in sorted(os.listdir(dir)):
        if 'rep' not in chrom_rep:
            continue
        rep = chrom_rep[-1]
        y = np.load(osp.join(dir, f'{chrom_rep}/chr{chrom}/y.npy'))
        if y_combined is None:
            y_combined = y.copy()
        else:
            y_combined += y
        # print(np.sum(y.diagonal() > 0))
        # y /= np.mean(y.diagonal(offset=1))
        # y[y==0] = np.NaN
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-{rep}')

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_combined, 'prob')
    plt.plot(meanDist, label = f'chr{chrom}-combined', c='k')
    plot_matrix(y_combined, osp.join(dir, f'y_chr{chrom}_combined.png'), vmax='mean')

    norm_file = osp.join(dir, f'chr{chrom}_multiHiCcompare.txt')
    if osp.exists(norm_file):
        y = np.loadtxt(norm_file)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-norm-combined', c='k', ls=':')
        # plot_matrix(y, osp.join(dir, f'y_chr{chrom}_norm.png'), vmax=np.mean(y_combined))


    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(2e-5, None)
    plt.ylabel('Contact Probability', fontsize=16)
    plt.xlabel('Polymer Distance (beads)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'p_s_chrom{chrom}.png'))
    plt.close()

def plot_p_s_chrom_norm(cell_line, chrom):
    '''Plots normalized replicates vs combined normalized replicate.'''
    dir = f'/home/erschultz/dataset_{cell_line}'
    for chrom_rep in sorted(os.listdir(dir)):
        if 'rep' not in chrom_rep:
            continue
        result = load_import_log(osp.join(dir, f'{chrom_rep}/chr{chrom}'))
        print(result)

        rep = chrom_rep[-1]
        y = np.loadtxt(osp.join(dir, f'{chrom_rep}/chr{chrom}/y_multiHiCcompare.txt'))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist, label = f'chr{chrom}-{result["cell_line"]}')

    # norm_file = osp.join(dir, f'chr{chrom}_multiHiCcompare.txt')
    # if osp.exists(norm_file):
    #     y = np.loadtxt(norm_file)
    #     meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    #     plt.plot(meanDist, label = f'chr{chrom}-norm', c='k', ls=':')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Contact Probability', fontsize=16)
    plt.xlabel('Polymer Distance (beads)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'p_s_chrom{chrom}_norm.png'))
    plt.close()


def plot_p_s_replicates_norm(cell_line, chrom):
    dir = f'/home/erschultz/dataset_{cell_line}'
    for chrom_rep in sorted(os.listdir(dir)):
        if 'rep' not in chrom_rep:
            continue
        rep = chrom_rep[-1]
        rep_dir = osp.join(dir, f'{chrom_rep}/chr{chrom}')
        y = np.loadtxt(osp.join(rep_dir, 'y_multiHiCcompare.txt'))
        plot_matrix(y, osp.join(rep_dir, 'y.png'), vmax='mean')
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        plt.plot(meanDist)

        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Contact Probability', fontsize=16)
        plt.xlabel('Polymer Distance (beads)', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(rep_dir, f'p_s_chrom{chrom}_norm.png'))
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

def compare_pc1():
    dir = '/home/erschultz/dataset_interp_test'
    for s in range(1, 8):
        for samples, name in zip(['samples', 'samples_pool'], ['interp', 'pool']):
            s_dir = osp.join(dir, f'{samples}/sample{s}')
            y = np.load(osp.join(s_dir, 'y.npy'))
            oe = epilib.get_oe(y)
            corr = np.corrcoef(oe)

            plot_matrix(corr, osp.join(s_dir, 'y_corr.png'), vmin='center', cmap='bluered')

            seqs = epilib.get_pcs(oe, 1, normalize = True)
            plt.plot(seqs, label=name)
            plt.xlabel('Beads', fontsize=16)
        plt.savefig(osp.join(dir, f'sample{s}_seqs.png'))
        plt.close()


if __name__ == '__main__':
    # compare_inpt_out_Lyu()
    # plot_chrom('11_20_23', 17)
    # plot_p_s_chrom('gm12878_variants', 17)
    plot_p_s_chrom_norm('11_17_23', 17)
    # plot_p_s_chroms('11_17_23')
    # plot_p_s_replicates_norm('11_17_23', 17)
    # compare_pc1()
