import os.path as osp

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from neural_net_utils.utils import make_dataset, InteractionConverter
from plotting_functions import plotContactMap

### Plotting contact frequency histograms
def getFrequencies(dataFolder, preprocessing, m, k, chi = None, save = True):
    '''
    Calculates number of times each interaction frequency value was observed
    accross samples in dataFolder.

    Inputs:
        dataFolder: location of input data`
        preprocessing: {'none', 'diag', 'diag_instance'}
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

    if preprocessing == 'none':
        y_file = 'y.npy'
    elif preprocessing == 'diag':
        y_file = 'y_diag.npy'
    elif preprocessing == 'diag_instance':
        y_file = 'y_diag_instance.npy'
    else:
        raise Exception("Invalid preprocessing", preprocessing)

    converter = InteractionConverter(k)
    samples = make_dataset(dataFolder)
    freq_arr = np.zeros((int(m * (m+1) / 2 * len(samples)), 4)) # freq, sample, type, psi_ij
    ind = 0
    for sample in samples:
        sampleid = int(osp.split(sample)[1][6:])

        x = np.load(osp.join(sample, 'x.npy'))

        y_path = osp.join(sample, y_file)
        if osp.exists(y_path):
            y = np.load(y_path)
        else:
            print("Warning: {} not found for sample {} - aborting".format(y_file, sample))
            return None

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
def genomic_distance_statistics(y, mode = 'freq', stat = 'mean'):
    '''
    Calculates statistics of contact frequency/probability as a function of genomic distance
    (i.e. along a give diagonal)

    Inputs:
        mode: freq for frequencies, prob for probabilities
        stat: mean to calculate mean, var for variance

    Outputs:
        result: numpy array where result[d] is the contact frequency/probability stat at distance d
    '''
    if mode == 'prob':
        y = y.copy() / np.max(y)

    if stat == 'mean':
        npStat = np.mean
    elif stat == 'var':
        npStat = np.var
    n = len(y)
    distances = range(0, n, 1)
    result = np.zeros_like(distances).astype(float)
    for d in distances:
        result[d] = npStat(np.diagonal(y, offset = d))

    return result

def plot_genomic_distance_statistics_inner(datafolder, diag, ofile, mode = 'freq', stat = 'mean'):
    """
    Function to plot expected interaction frequency as a function of genomic distance for all samples in dataFolder.

    Inputs:
        dataFolder: location of data to plot
        ofile: save location
        title: plot title, None for no title
        mode: freq for frequencies, prob for probabilities
        stat: mean to calculate mean, var for variance
    """
    fig, ax = plt.subplots()
    samples = make_dataset(datafolder)
    for sample in samples:
        if diag:
            y_diag_path = osp.join(sample, 'y_diag.npy')
            if osp.exists(y_diag_path):
                y = np.load(y_diag_path)
            else:
                print("Warning: no diag processing found")
                return
        else:
            y = np.load(osp.join(sample, 'y.npy'))
        result = genomic_distance_statistics(y, mode = mode, stat = stat)
        ax.plot(result, label = osp.split(sample)[1])
    if not diag:
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
    if diag:
        plt.title('diag preprocessing', fontsize = 16)
    else:
        plt.title('no preprocessing', fontsize = 16)
    plt.legend()
    plt.savefig(ofile)
    plt.close()

def plot_genomic_distance_statistics(dataFolder):
    '''Wrapper function for plot_genomic_distance_statistics_inner.'''
    for diag in [True, False]:
        for stat in ['mean', 'var']:
            ofile = osp.join(dataFolder, "freq_stat_{}_diag_{}.png".format(stat, diag))
            plot_genomic_distance_statistics_inner(dataFolder, diag, ofile, stat = stat)

### basic plots
def basic_plots(dataFolder, plot_y = True, plot_s = True, chi = None):
    '''Generate basic plots of data in dataFolder.'''
    if chi is None:
        chi_path = osp.join(dataFolder, 'chis.npy')
        if osp.exists(chi_path):
            chi = np.load(chi_path)

    in_paths = sorted(make_dataset(dataFolder))
    for path in in_paths:
        print(path)
        if plot_y:
            y = np.load(osp.join(path, 'y.npy'))
            plotContactMap(y, osp.join(path, 'y.png'), title = 'pre normalization', vmax = 'mean')

            y_diag_path = osp.join(path, 'y_diag.npy')
            if osp.exists(y_diag_path):
                y_diag = np.load(y_diag_path)
                plotContactMap(y_diag, osp.join(path, 'y_diag.png'), title = 'diag normalization', vmax = 'max')

            y_diag_instance_path = osp.join(path, 'y_diag_instance.npy')
            if osp.exists(y_diag_instance_path):
                y_diag_instance = np.load(y_diag_instance_path)
                plotContactMap(y_diag_instance, osp.join(path, 'y_diag_instance.png'), title = 'diag instance normalization', vmax = 'max')

            y_prcnt_path = osp.join(path, 'y_prcnt.npy')
            if osp.exists(y_prcnt_path):
                y_prcnt = np.load(y_prcnt_path)
                plotContactMap(y_prcnt, osp.join(path, 'y_prcnt.png'), title = 'prcnt normalization', vmax = 'max', prcnt = True)

        if plot_s:
            x = np.load(osp.join(path, 'x.npy'))
            if chi is None:
                chi_path = osp.join(path, 'chis.npy')
                if osp.exists(chi_path):
                    chi = np.load(chi_path)
                else:
                    raise Exception('chi not found at ', chi_path)
            s = x @ chi @ x.T
            plotContactMap(s, osp.join(path, 's.png'), title = 'S', vmax = 'max', vmin = 'min', cmap = 'blue-red')


if __name__ == '__main__':
    dataset = 'dataset_10_27_21'
    sample = 40
    basic_plots(dataset, plot_y = False)
    # plot_genomic_distance_statistics(dataset)
    # freqSampleDistributionPlots(dataset, sample, splits = [None])
