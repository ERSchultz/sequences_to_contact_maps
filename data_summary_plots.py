import json
import math
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dataset_classes import make_dataset
from utils.InteractionConverter import InteractionConverter
from utils.load_utils import load_all, load_X_psi, load_Y
from utils.plotting_utils import (plot_matrix, plot_seq_binary,
                                  plot_seq_exclusive)
from utils.utils import DiagonalPreprocessing


def chi_to_latex(chi, ofile):
    # TODO
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

### Plotting contact frequency histograms
def getFrequencies(dataFolder, preprocessing, m, k, chi = None, save = True):
    '''
    Calculates number of times each interaction frequency value was observed
    accross samples in dataFolder.

    Inputs:
        dataFolder: location of input data
        preprocessing: {None, 'none', 'diag'}
        m: numper of particles
        k: number of particle types
        chi: chi matrix
        save: True to save result as .npy

    Output:
        freq_arr: 2D np array containing all observed contact frequencies

    freq_arr[:, 0] is frequency
    freq_arr[:, 1] is sample
    freq_arr[:, 2] is type
    freq_arr[:, 3] is psi_ij
    '''
    freq_path = osp.join(dataFolder, 'freq_arr_preprocessing_{}.npy'.format(preprocessing))
    if osp.exists(freq_path):
        return np.load(freq_path)

    if preprocessing not in {None, 'none', 'diag'}:
        raise Exception(f"Invalid preprocessing: {preprocessing}")

    converter = InteractionConverter(k)
    samples = make_dataset(dataFolder)
    freq_arr = np.zeros((int(m * (m+1) / 2 * len(samples)), 4)) # freq, sample, type, psi_ij
    ind = 0
    for sample in samples:
        sampleid = int(osp.split(sample)[1][6:])

        x, psi = load_X_psi(sample)

        y, ydiag = load_Y(sample)
        if preprocessing == 'diag':
            y = ydiag

        for i in range(m):
            xi = x[i]
            for j in range(i+1):
                xj = x[j]
                comb = frozenset({tuple(xi), tuple(xj)})
                comb_type = converter.comb2Type(comb)
                if chi is not None:
                    psi_ij = xi @ chi @ xj
                else:
                    psi_ij = None
                freq_arr[ind] = [y[i,j], sampleid, comb_type, psi_ij]
                ind += 1

    if save:
        np.save(freq_path, freq_arr)
    return freq_arr

def plotFrequenciesForSample(freq_arr, dataFolder, preprocessing, k, sampleid, split = 'type', xmax = None, log = False):
    """
    Plotting function for frequency distributions corresponding to only one sample.

    Inputs:
        freq_arr: numpy array with 4 columns (freq, sampleid, interaction_type, psi)
        dataFolder: location of data, used for saving image
        preprocessing: type of preprocessing {'None', 'diag', 'diag_instance'}
        sampleid: int, which sample id to plot
        k: number of epigentic marks, used for InteractionConverter
        split: how to split data into subplots: None for no subplots, type for subplot on interaction type, psi for sublot on interation psi
        xmax: x axis limit
    """
    freq_pd = pd.DataFrame(freq_arr, columns = ['freq', 'sampleID', 'type', 'psi']) # cols = freq, sampleid, interaction_type
    if k is not None:
        converter = InteractionConverter(k)

    if split is None:
        fig = plt.figure(figsize=(8, 4))
    elif split == 'type':
        fig = plt.figure(figsize=(12, 12))
    elif split == 'psi':
        fig = plt.figure(figsize=(10, 5))
    bigax = fig.add_subplot(111, label = 'bigax')
    indplt = 1
    for g_name, g_df in freq_pd.groupby(['sampleID']):
        if g_name == sampleid:
            if split is None:
                ax = fig.add_subplot(1, 1, indplt)
                if log:
                    ax.hist(np.log10(g_df['freq']), bins = 200)
                else:
                    ax.hist(g_df['freq'], bins = 200)
                ax.set_yscale('log')
                indplt += 1
            elif split == 'type':
                for g_name_2, g_df_2 in g_df.groupby(['type']):
                    ax = fig.add_subplot(5, 2, indplt)
                    ax.hist(g_df_2['freq'], bins = 100)
                    ax.set_title(converter.comb2str(converter.type2Comb(g_name_2)))
                    ax.set_yscale('log')
                    indplt += 1
            elif split == 'psi':
                num_plots = len(g_df.groupby(['psi']))
                cols = 4
                rows = math.ceil(num_plots / cols)
                for g_name_2, g_df_2 in g_df.groupby(['psi']):
                    ax = fig.add_subplot(rows, cols, indplt)
                    ax.hist(g_df_2['freq'], bins = 100)
                    ax.set_title(g_name_2)
                    ax.set_yscale('log')
                    indplt += 1

    # Turn off axis lines and ticks of the big subplot
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor = 'w', top = False, bottom = False, left = False, right = False)
    # set axis labels
    if preprocessing == 'none':
        bigax.set_xlabel('contact frequency', fontsize = 16)
    else:
        bigax.set_xlabel('observed/expected contact frequency', fontsize = 16)
    bigax.set_ylabel('count ', fontsize = 16)

    fig.tight_layout()
    # if preprocessing == 'none':
    #     fig.suptitle('sample{}'.format(sampleid), fontsize = 16, y = 1)
    # else:
    #     fig.suptitle('sample{} {} preprocessing'.format(sampleid, preprocessing), fontsize = 16, y = 1)

    if xmax is not None:
        plt.xlim(right = xmax)

    f_name = 'freq_count'
    if split is not None:
        f_name += '_split_{}'.format(split)
    if preprocessing != 'none':
        f_name += '_' + preprocessing
    if log:
        f_name += '_log'
    f_path = osp.join(dataFolder, 'samples', "sample{}".format(sampleid), f_name + '.png')
    plt.savefig(f_path)
    plt.close()

def plotFrequenciesSampleSubplot(freq_arr, dataFolder, preprocessing, k, split = 'type'):
    """
    Plotting function for frequency distributions where each subplot corresponds to one sample.

    Inputs:
        freq_arr: numpy array with 4 columns (freq, sampleid, interaction_type, psi)
        dataFolder: location of data, used for saving image
        preprocessing: type of preprocessing {'None', 'diag', 'diag_instance'}
        k: number of epigentic marks, used for InteractionConverter
        split: how to split data within each subplot: None for no split, type for split on interaction type, psi for split on interation psi
    """
    freq_pd = pd.DataFrame(freq_arr, columns = ['freq', 'sampleID', 'type', 'psi']) # cols = freq, sampleid, interaction_type, psi
    converter = InteractionConverter(k)

    fig = plt.figure(figsize=(12, 12))
    bigax = fig.add_subplot(111, label = 'bigax') # use bigax to set overall axis labels
    indplt = 1
    for g_name, g_df in freq_pd.groupby(['sampleID']):
        if indplt < 10: # only plot first 9 samples, design choice for convenient implemenation
            ax = fig.add_subplot(3, 3, indplt)
            if split is None:
                ax.hist(g_df['freq'], bins = 100)
            elif split.lower() == 'type':
                for g_name_2, g_df_2 in g_df.groupby(['type']):
                    ax.hist(g_df_2['freq'], label = converter.comb2str(converter.type2Comb(g_name_2)), bins = 100)
            elif split.lower() == 'psi':
                for g_name_2, g_df_2 in g_df.groupby(['psi']):
                    ax.hist(g_df_2['freq'], label = g_name_2, bins = 100)
            ax.set_title('sample{}'.format(int(g_name)))
            ax.set_yscale('log')
            indplt += 1

    # Turn off axis lines and ticks on bigax
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor = 'w', top = False, bottom = False, left = False, right = False)
    # set axis labels on bigax
    bigax.set_xlabel('contact frequency', fontsize = 16)
    bigax.set_ylabel('count of contact frequency', fontsize = 16)

    if split is not None:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'upper right', title = split)

    fig.tight_layout()
    if preprocessing == 'none':
        fig.suptitle('no preprocessing', fontsize = 16, y = 1)
    else:
        fig.suptitle('{} preprocessing'.format(preprocessing), fontsize = 16, y = 1)

    plt.savefig(osp.join(dataFolder, 'freq_count_multisample_preprocessing_{}_split_{}.png'.format(preprocessing, split)))
    plt.close()

def freqSampleDistributionPlots(dataFolder, sample_id, m = 1024, k = None, splits = [None, 'psi']):
    '''Wrapper function for plotFrequenciesForSample and plotFrequenciesSampleSubplot.'''
    chi_path1 = osp.join(dataFolder, 'chis.npy')
    chi_path2 = osp.join(dataFolder, 'samples/sample{}'.format(sample_id), 'chis.npy')
    if osp.exists(chi_path1):
        chi = np.load(chi_path1)
        k = len(chi)
    elif osp.exists(chi_path2):
        chi = np.load(chi_path2)
        k = len(chi)
    else:
        chi = None
        assert k is not None, "need to input k if chi_path is missing"

    # freq distribution plots
    for preprocessing in ['none', 'diag', 'diag_instance']:
        print(preprocessing)
        freq_arr = getFrequencies(dataFolder, preprocessing, m, k, chi)
        if freq_arr is None:
            continue

        for split in splits:
            print('\t', split)
            # plotFrequenciesSampleSubplot(freq_arr, dataFolder, preprocessing, k, split)
            plotFrequenciesForSample(freq_arr, dataFolder, preprocessing, k, sampleid = sample_id, split = split)

### Plotting contact frequency as function of genomic distance
def plot_genomic_distance_statistics_inner(datafolder, ifile, ofile, title,  mode = 'freq', stat = 'mean'):
    """
    Function to plot expected interaction frequency as a function of genomic distance for all samples in dataFolder.

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

def plot_mean_vs_genomic_distance(y, path, diag_chis):
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                            zero_diag = True, zero_offset = 3)
    plt.plot(meanDist)
    plt.yscale('log')

    if diag_chis is not None:
        m = len(y)
        k = len(diag_chis)
        bin_size = m / k

        # find chi transition points
        transitions = [0]
        prev_diag_chi = diag_chis[0]
        for i in range(1, m):
            diag_chi = diag_chis[math.floor(i/(m/k))]
            if diag_chi != prev_diag_chi:
                transitions.append(i)
                prev_diag_chi = diag_chi

        start = np.log10(np.max(meanDist)*1.05)
        stop = np.log10(np.max(meanDist[-int(0.9*k):])*1.05)
        annotate_y_arr = np.logspace(start, stop, k)
        for i, diag_chi, annotate_y in zip(transitions, diag_chis, annotate_y_arr):
            plt.axvline(i, linestyle = 'dashed', color = 'green')
            plt.annotate(f'{np.round(diag_chi, 1)}', (i, annotate_y))

    plt.ylabel('Contact Probability', fontsize = 16)
    plt.xlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(path, 'meanDist.png'))
    plt.close()


### basic plots
def basic_plots(dataFolder, plot_y = False, plot_energy = True, plot_x = True, plot_chi = False, sampleID = None):
    '''Generate basic plots of data in dataFolder.'''
    in_paths = sorted(make_dataset(dataFolder))
    for path in in_paths:
        if isinstance(sampleID, list) and int(osp.split(path)[1][6:]) not in sampleID:
            continue
        elif isinstance(sampleID, int) and osp.split(path)[1] != f'sample{sampleID}':
            continue
        print(path)

        x, psi, chi, chi_diag, e, s, y, ydiag = load_all(path, data_folder = dataFolder,
                                                save = True,
                                                throw_exception = False)

        if plot_y:
            plot_matrix(y, osp.join(path, 'y.png'), vmax = 'mean')
            plot_matrix(ydiag, osp.join(path, 'y_diag.png'), title = 'diag normalization', vmax = 'max')
            y_log = np.log(y + 1e-8)
            plot_matrix(y_log, osp.join(path, 'y_log.png'), title = 'log normalization', vmax = 'max')

            plot_mean_vs_genomic_distance(y, path, chi_diag)

            y_prcnt_path = osp.join(path, 'y_prcnt.npy')
            if osp.exists(y_prcnt_path):
                y_prcnt = np.load(y_prcnt_path)
                plot_matrix(y_prcnt, osp.join(path, 'y_prcnt.png'), title = 'prcnt normalization', vmax = 'max', prcnt = True)

            y_diag_batch_path = osp.join(path, 'y_diag_batch.npy')
            if osp.exists(y_diag_batch_path):
                y_diag_batch = np.load(y_diag_batch_path)
                plot_matrix(y_diag_batch, osp.join(path, 'y_diag_batch.png'), title = 'diag normalization', vmax = 'max')

        if chi is not None:
            chi_to_latex(chi, ofile = osp.join(path, 'chis.tek'))
            if plot_chi:
                plot_matrix(chi, osp.join(path, 'chi.png'), vmax = 'max', vmin = 'min', cmap = 'blue-red')

        if chi_diag is not None:
            plt.plot(chi_diag)
            plt.xlabel('Bin', fontsize = 16)
            plt.ylabel('Diagonal Parameter', fontsize = 16)
            plt.savefig(osp.join(path, 'chi_diag.png'))
            plt.close()

        if plot_energy:
            if s is not None:
                plot_matrix(s, osp.join(path, 's.png'), vmax = 'max', vmin = 'min', cmap = 'blue-red')
                s_sym = (s + s.T)/2
                np.save(osp.join(path, 's_sym.npy'), s_sym)
                plot_matrix(s, osp.join(path, 's_sym.png'), vmax = 'max', vmin = 'min', cmap = 'blue-red')

            if e is not None:
                plot_matrix(e, osp.join(path, 'e.png'), vmax = 'max', vmin = 'min', cmap = 'blue-red')

        if plot_x:
            x_path = osp.join(path, 'x.npy')
            if osp.exists(x_path):
                x = np.load(x_path)
            else:
                continue

            plot_seq_binary(x, ofile = osp.join(path, 'x.png'))
            # plot_seq_exclusive(x, ofile = osp.join(path, 'x.png'))

if __name__ == '__main__':
    dir = '/project2/depablo/erschultz'
    dir = '/home/erschultz/sequences_to_contact_maps'
    dir = '/home/erschultz'

    dataset = 'dataset_test_diag'
    data_dir = osp.join(dir, dataset)
    basic_plots(data_dir, plot_y = True, plot_energy = False, plot_x = False,
                    sampleID = [140, 141])
    # plot_genomic_distance_statistics(data_dir)
    # freqSampleDistributionPlots(dataset, sample, splits = [None])
    # getPairwiseContacts(data_dir)
