import json
import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scripts.energy_utils import calculate_D, s_to_E
from scripts.InteractionConverter import InteractionConverter
from scripts.knightRuiz import knightRuiz
from scripts.load_utils import load_all, load_contact_map, load_X_psi, load_Y
from scripts.neural_nets.dataset_classes import make_dataset
from scripts.plotting_utils import (plot_diag_chi, plot_matrix,
                                    plot_mean_vs_genomic_distance,
                                    plot_seq_binary, plot_seq_continuous)
from scripts.utils import DiagonalPreprocessing
from sklearn.metrics import mean_squared_error


def chi_to_latex(chi, ofile):
    with open(ofile, 'w') as f:
        f.write('\\begin{bmatrix}\n')
        for row in range(len(chi)):
            chi_row = [str(np.round(i, 1)) for i in chi[row]]
            f.write(' & '.join(chi_row) + ' \\\ \n')
        f.write('\\end{bmatrix}\n')

def getPairwiseContacts(data_folder):
    '''
    Calculates number of pairs of each type of label in psi.
    Used to check if some labels are too rare, which could cause problems.'
    '''
    samples = make_dataset(data_folder)
    for sample in samples:
        print(sample)
        _, psi = load_X_psi(sample)
        m, k = psi.shape
        label_count = np.sum(psi, axis = 0)
        print(label_count)

        label_pair_count = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                val = np.sum(np.logical_and(psi[:, i] == 1, psi[:, j] == 1))
                label_pair_count[i, j] = val
        print(label_pair_count)

        print('')

### Plotting contact frequency as function of genomic distance
def plot_genomic_distance_statistics_inner(datafolder, ifile, ofile, title,
                                            mode = 'freq', stat = 'mean'):
    """
    Function to plot expected interaction frequency as a function of genomic distance
    for all samples in dataFolder.

    Inputs:
        dataFolder: location of data to plot
        ifile: name of y file to load
        ofile: save location
        title: plot title, None for no title
        mode: freq for frequencies, prob for probabilities
        stat: mean to calculate mean, var for variance
    """
    fig, ax = plt.subplots()
    samples = make_dataset(datafolder)
    mean_result = []
    for i, sample in enumerate(samples):
        if i > 10:
            break
        y = np.load(osp.join(sample, ifile))
        result = DiagonalPreprocessing.genomic_distance_statistics(y, mode = mode, stat = stat)[10:]
        mean_result.append(result)
        ax.plot(result, label = osp.split(sample)[1])
    mean_result = np.mean(mean_result, axis = 0)
    ax.plot(mean_result, label = 'mean', color = 'k')
    if 'diag' not in ifile:
        plt.yscale('log')

    mode_string = ''
    if mode == 'freq':
        mode_string = 'frequency'
    elif mode == 'prob':
        mode_string = 'probability'

    stat_string = ''
    if stat == 'mean':
        stat_string = 'mean'
    elif stat == 'var':
        stat_string = 'variance of'
    plt.ylabel('{} contact {}'.format(stat_string, mode_string), fontsize = 16)
    plt.xlabel('distance', fontsize = 16)
    if title is not None:
        plt.title(title, fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def plot_genomic_distance_statistics(dataFolder):
    '''Wrapper function for plot_genomic_distance_statistics_inner.'''
    for title in ['y', 'y_diag']:
        for stat in ['mean', 'var']:
            ifile = title + '.npy'
            ofile = osp.join(dataFolder, f"freq_stat_{stat}_{title}.png")
            plot_genomic_distance_statistics_inner(dataFolder, ifile, ofile,
                                                    title, stat = stat)

def compare_kr(path, inp, ref, ref_diag, inp_name = 'y', ref_name = 'y'):
    ref_p = ref / np.mean(np.diagonal(ref))
    p_mean = np.mean(ref_p)

    inp_kr_file = osp.join(path, f'{inp_name}_kr.npy')
    if osp.exists(inp_kr_file):
        inp_kr = np.load(inp_kr_file)
    else:
        print(np.count_nonzero(np.isnan(inp)))
        inp_kr = knightRuiz(inp)
        print(inp_kr)
        np.save(inp_kr_file, inp_kr)
    plot_matrix(inp_kr, osp.join(path, f'{inp_name}_kr.png'), title = 'kr normalization', vmax = 'mean')

    inp_kr_p = inp_kr/np.mean(np.diagonal(inp_kr))
    plot_matrix(inp_kr_p, osp.join(path, f'{inp_name}_kr_p.png'), title = 'kr normalization', vmax = p_mean)

    diff = ref_p - inp_kr_p
    plot_matrix(diff, osp.join(path, f'{inp_name}_kr_p_vs_{ref_name}_p.png'),
                title = f'{ref_name}_p - {inp_name}_kr_p', cmap = 'blue-red')

    meanDist_inp_kr = DiagonalPreprocessing.genomic_distance_statistics(inp_kr)
    inp_kr_diag = DiagonalPreprocessing.process(inp_kr, meanDist_inp_kr)
    plot_matrix(inp_kr_diag, osp.join(path, f'{inp_name}_kr_diag.png'),
                title = 'kr + diag normalization', vmin = 'center1', cmap = 'blue-red')

    diff = ref_diag - inp_kr_diag
    plot_matrix(diff, osp.join(path, f'{inp_name}_kr_diag_vs_{ref_name}_diag.png'),
                title = f'{ref_name}_diag - {inp_name}_kr_diag', vmin = 'center', cmap = 'blue-red')

    return inp_kr, inp_kr_diag

def compare_sweep(ref, ref_diag, path, figures_path, ref_name = 'y'):
    ref_p = ref / np.mean(np.diagonal(ref))

    for sweep in [100000, 200000, 300000, 400000, 500000]:
        y_ifile = osp.join(path, f'data_out/contacts{sweep}.txt')
        y_ofile = osp.join(figures_path, f'y_sweep{sweep}.npy')
        y_diag_ofile =  osp.join(figures_path, f'y_sweep{sweep}_diag.npy')
        if osp.exists(y_ofile):
            y_sweep = np.load(y_ofile)
        elif osp.exists(y_ifile):
            y_sweep = np.loadtxt(y_ifile)
            np.save(y_ofile, y_sweep)
            plot_matrix(y_sweep, osp.join(figures_path, f'y_sweep{sweep}.png'), vmax='mean')
        sweep_p = y_sweep / np.mean(np.diagonal(y_sweep))
        diff = ref_p - sweep_p
        plot_matrix(diff, osp.join(figures_path, f'p_sweep{sweep}_vs_{ref_name}_p.png'),
                    title = f'{ref_name}_p - sweep{sweep}_p',
                    vmin='center', cmap = 'blue-red')

        compare_kr(figures_path, y_sweep, ref, ref_diag, f'y_sweep{sweep}', ref_name)

        if not osp.exists(y_diag_ofile):
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_sweep)
            y_sweep_diag = DiagonalPreprocessing.process(y_sweep, meanDist)
            np.save(y_diag_ofile, y_sweep_diag)
        else:
            y_sweep_diag = np.load(y_diag_ofile)

        plot_matrix(y_sweep_diag, osp.join(figures_path, f'y_sweep{sweep}_diag.png'),
                    vmin='center1', cmap='blue-red')

        diff = ref_diag - y_sweep_diag
        plot_matrix(diff, osp.join(figures_path, f'y_sweep{sweep}_diag_vs_{ref_name}_diag.png'),
                        title = f'{ref_name}_diag - y_sweep{sweep}_diag',
                        vmin='center', cmap = 'blue-red')


### basic plots
def basic_plots(dataFolder, plot_y = False, plot_energy = True, plot_x = True,
                plot_chi = False, sampleID = None):
    '''Generate basic plots of data in dataFolder.'''
    in_paths = sorted(make_dataset(dataFolder, use_ids = False))
    for path in in_paths:
        id = osp.split(path)[1][6:]
        if isinstance(sampleID, list) and id not in sampleID:
            continue
        elif isinstance(sampleID, int):
            if not id.isnumeric():
                continue
            elif int(id) != sampleID:
                continue
        print(path)

        figures_path = osp.join(path, 'figures')
        if not osp.exists(figures_path):
            os.mkdir(figures_path, mode = 0o755)

        x, psi, chi, _, e, s, y, ydiag = load_all(path, data_folder = dataFolder,
                                                save = True,
                                                throw_exception = False)

        config_file = osp.join(path, 'config.json')
        config = None
        dense = False
        cutoff = None
        loading = None
        if osp.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            if 'dense_diagonal_on' in config.keys():
                dense = config['dense_diagonal_on']
            if 'dense_diagonal_cutoff' in config.keys():
                cutoff = config['dense_diagonal_cutoff']
            if 'dense_diagonal_loading' in config.keys():
                loading = config['dense_diagonal_loading']

        if plot_y:
            contacts = int(np.sum(y) / 2)
            sparsity = np.round(np.count_nonzero(y) / len(y)**2 * 100, 2)
            title = f'# contacts: {contacts}, sparsity: {sparsity}%'
            plot_matrix(y, osp.join(path, 'y.png'), vmax = 'mean', title = title)
            np.savetxt(osp.join(path, 'y.txt'), y)

            meanDist = plot_mean_vs_genomic_distance(y, path, 'meanDist.png', config = config)
            np.savetxt(osp.join(path, 'meanDist.txt'), meanDist)
            plot_mean_vs_genomic_distance(y, path, 'meanDist_log.png', logx = True, config = config)

            p = y/np.mean(np.diagonal(y))
            plot_matrix(p, osp.join(path, 'p.png'), vmax = 'mean')

            plot_matrix(ydiag, osp.join(path, 'y_diag.png'),
                        title = f'Sample {id}\ndiag normalization', vmin = 'center1', cmap = 'blue-red')
            ydiag_log = np.log(ydiag)
            ydiag_log[np.isinf(ydiag_log)] = 0
            nonzero = np.count_nonzero(ydiag_log)
            prcnt = np.round(nonzero / 1024**2 * 100, 1)
            plot_matrix(ydiag_log, osp.join(path, 'y_diag_log.png'), title = f'diag + log\nEdges={nonzero} ({prcnt}%)', cmap = 'bluered')


            y_kr, y_kr_diag = compare_kr(figures_path, y, y, ydiag)
            plot_mean_vs_genomic_distance(y_kr, figures_path, 'meanDistKR_log.png',
                                        ref = meanDist, ref_label = 'not KR', logx = True)


            # ydiag_log_sparse = ydiag_log.copy()
            # ydiag_log_sparse[np.abs(ydiag_log_sparse) < 0.405] = 0
            # nonzero = np.count_nonzero(ydiag_log_sparse)
            # prcnt = np.round(nonzero / 1024**2 * 100, 1)
            # plot_matrix(ydiag_log_sparse, osp.join(path, 'ydiag_log_sparse.png'), title = f'diag + log + sparse\nEdges={nonzero} ({prcnt}%)', cmap = 'bluered')

            # y_diag_log_triu = ydiag_log.copy()
            # y_diag_log_triu = np.tril(y_diag_log_triu, 512)
            # y_diag_log_triu = np.triu(y_diag_log_triu, -512)
            # nonzero = np.count_nonzero(y_diag_log_triu)
            # prcnt = np.round(nonzero / 1024**2 * 100, 1)
            # plot_matrix(y_diag_log_triu, osp.join(path, 'y_diag_log_triu.png'),
            #             title = f'diag + log + triu normalization\nEdges={nonzero} ({prcnt}%)', cmap = 'bluered')
            #
            # y_diag_log_sparse_triu = ydiag_log_sparse.copy()
            # y_diag_log_sparse_triu = np.tril(y_diag_log_sparse_triu, 512)
            # y_diag_log_sparse_triu = np.triu(y_diag_log_sparse_triu, -512)
            # nonzero = np.count_nonzero(y_diag_log_sparse_triu)
            # prcnt = np.round(nonzero / 1024**2 * 100, 1)
            # plot_matrix(y_diag_log_sparse_triu, osp.join(path, 'y_diag_log_sparse_triu.png'),
            #             title = f'diag + log + sparse + triu\nEdges={nonzero} ({prcnt}%)', cmap = 'bluered')


            y_log_file = osp.join(path, 'y_log.npy')
            if osp.exists(y_log_file):
                y_log = np.load(y_log_file)
            else:
                y_log = np.log(y + 1)
            plot_matrix(y_log, osp.join(path, 'y_log.png'), title = 'log normalization', vmax = 'max')
            plot_mean_vs_genomic_distance(y_log, figures_path, 'meanDistLog.png', config = config)

            y_log_diag_file = osp.join(path, 'y_log_diag.npy')
            if osp.exists(y_log_diag_file):
                y_log_diag = np.load(y_log_diag_file)
            else:
                meanDistLog = DiagonalPreprocessing.genomic_distance_statistics(y_log)
                np.savetxt(osp.join(path, 'meanDistLog.txt'), meanDistLog)
                y_log_diag = DiagonalPreprocessing.process(y_log, meanDistLog)
            nonzero = np.count_nonzero(y_log_diag)
            prcnt = np.round(nonzero / 1024**2 * 100, 1)
            plot_matrix(y_log_diag, osp.join(path, 'y_log_diag.png'),
                        title = f'log + diag\nEdges={nonzero} ({prcnt}%)')

            y_log_diag_log = np.log(y_log_diag)
            y_log_diag_log[np.isinf(y_log_diag_log)] = 0
            nonzero = np.count_nonzero(y_log_diag_log)
            prcnt = np.round(nonzero / 1024**2 * 100, 1)
            plot_matrix(y_log_diag_log, osp.join(path, 'y_log_diag_log.png'),
                        title = f'log + diag + log\nEdges={nonzero} ({prcnt}%)', vmax = 'max', cmap = 'bluered')

            # y_log_diag_log_sparse = y_log_diag_log.copy()
            # y_log_diag_log_sparse[np.abs(y_log_diag_log_sparse) < 0.405] = 0
            # nonzero = np.count_nonzero(y_log_diag_log_sparse)
            # prcnt = np.round(nonzero / 1024**2 * 100, 1)
            # plot_matrix(y_log_diag_log_sparse, osp.join(path, 'y_log_diag_log_sparse.png'), title = f'log + diag + log + sparse\nEdges={nonzero} ({prcnt}%)', cmap = 'bluered')

            # compare_sweep(y, ydiag, path, figures_path)
            # compare_sweep(y_kr, y_kr_diag, path, figures_path, 'y_kr')

        if chi is not None:
            chi_to_latex(chi, ofile = osp.join(path, 'chis.tek'))
            if plot_chi:
                plot_matrix(chi, osp.join(path, 'chi.png'), vmax = 'max', vmin = 'min',
                            cmap = 'blue-red')

        # plot diag chi
        diag_chis_continuous_file = osp.join(path, 'diag_chis_continuous.npy')
        if config is not None:
            diag_chis_step = plot_diag_chi(config, path, ref = diag_chis_continuous_file, ref_label = 'continuous')
            D = calculate_D(diag_chis_step)
            plot_matrix(D, osp.join(path, 'D.png'), vmax = 'max', vmin = 0)


        if plot_energy:
            SD = None
            if s is not None:
                plot_matrix(s, osp.join(path, 's.png'), vmax = 'max', vmin = 'min',
                            cmap = 'blue-red')
                s_sym = (s + s.T)/2
                np.save(osp.join(path, 's_sym.npy'), s_sym)
                plot_matrix(s_sym, osp.join(path, 's_sym.png'), vmax = 'max', vmin = 'min',
                            cmap = 'blue-red')

                SD = s_sym + D + np.diag(np.diagonal(D.copy()))
                np.save(osp.join(path, 'sd.npy'), SD)
                SD_gnn = s_sym + D
                np.save(osp.join(path, 'sd_gnn.npy'), SD_gnn)
                plot_matrix(SD, osp.join(path, 'SD.png'), vmax = 'max', vmin = 'min',
                            cmap = 'blue-red')

                # ref = np.loadtxt(osp.join(path, 'GNN-223-S/k0/replicate1/resources/s_matrix.txt'))
                # ref_plaid = np.loadtxt(osp.join(path, 'GNN-223-S/k0/replicate1/resources/plaid_hat.txt'))
                # ref_diag = np.loadtxt(osp.join(path, 'GNN-223-S/k0/replicate1/resources/diagonal_hat.txt'))
                # print(mean_squared_error(SD, ref))
                # res = minimize(loss, (1, 1), args = (SD, ref_plaid, ref_diag))
                # print(res)

            if e is not None:
                plot_matrix(e, osp.join(path, 'e.png'), vmax = 'max', vmin = 'min',
                            cmap = 'blue-red')
                if SD is None:
                    ED = e + 2*D
                else:
                    ED = s_to_E(SD)
                np.save(osp.join(path, 'ed.npy'), ED)
                plot_matrix(ED, osp.join(path, 'ED.png'), vmax = 'max', vmin = 'min',
                            cmap = 'blue-red')

        if plot_x:
            x_path = osp.join(path, 'x.npy')
            if osp.exists(x_path):
                x = np.load(x_path)
            else:
                continue

            # plot_seq_binary(x, ofile = osp.join(path, 'x.png'))
            plot_seq_continuous(x,  ofile = osp.join(path, 'x.png'))
            # plot_seq_exclusive(x, ofile = osp.join(path, 'x.png'))

# def loss(params, a, b, c):
#     alpha, beta = params
#     return mean_squared_error(a, alpha*b+beta*c)

if __name__ == '__main__':
    dir = '/project2/depablo/erschultz'
    dir = '/home/erschultz/sequences_to_contact_maps'
    dir = '/home/erschultz'

    dataset = 'dataset_02_04_23'
    data_dir = osp.join(dir, dataset)
    basic_plots(data_dir, plot_y = True, plot_energy = False, plot_x = False,
                plot_chi = False, sampleID = 277)
    # plot_genomic_distance_statistics(data_dir)
    # freqSampleDistributionPlots(dataset, sample, splits = [None])
    # getPairwiseContacts(data_dir)
