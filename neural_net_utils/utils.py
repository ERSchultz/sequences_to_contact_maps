import torch
from torch.utils.data import DataLoader
from numba import jit, njit
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import argparse
import seaborn as sns
from collections import defaultdict
import pandas as pd

# dataset functions
def make_dataset(dir, minSample = 0):
    data_file_arr = []
    samples_dir = os.path.join(dir, 'samples')
    for file in os.listdir(samples_dir):
        if not file.startswith('sample'):
            print("Skipping {}".format(file))
        else:
            sample_id = int(file[6:])
            if sample_id < minSample:
                print("Skipping {}".format(file))
            else:
                data_file = os.path.join(samples_dir, file)
                data_file_arr.append(data_file)
    return data_file_arr

def getDataLoaders(dataset, batch_size, num_workers, seed, split = [0.8, 0.1, 0.1], shuffle = True):
    train_dataset, val_dataset, test_dataset = splitDataset(dataset, split, seed)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                                    shuffle = shuffle, num_workers = num_workers)
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size,
                                        shuffle = shuffle, num_workers = num_workers)
    else:
        val_dataloader = None
    if len(val_dataset) > 0:
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size,
                                        shuffle = shuffle, num_workers = num_workers)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader

def splitDataset(dataset, split, seed):
    """Splits input dataset into proportions specified by split."""
    N = len(dataset)
    assert sum(split) - 1 < 1e-5, "split doesn't sum to 1: {}".format(split)
    trainN = math.floor(N * split[0])
    valN = math.floor(N * split[1])
    testN = N - trainN - valN
    return torch.utils.data.random_split(dataset,
                                        [trainN, valN, testN],
                                        generator = torch.Generator().manual_seed(seed))

# data processing functions
@njit
def x2xx(x, mode = 'add'):
    # TODO better explanation here
    """
    Function for converting x to an image.

    Inputs:
        x: (n x k) array
        mode: method for combining xi and xj

    Outputs:
        xx: (2k x n x n) or (k x n x n) array
    """
    n, k = x.shape
    if mode == 'append':
        xx = np.zeros((k*2, n, n))
    else:
        xx = np.zeros((k, n, n))
    for i in range(n):
        for j in range(i+1):
            if mode == 'append':
                xx[:, i, j] = np.append(x[i], x[j])
                xx[:, j, i] = np.append(x[j], x[i])
            elif mode =='add':
                xx[:, i, j] = np.add(x[i], x[j])
                xx[:, j, i] = np.add(x[j], x[i])
            elif mode == 'mean':
                xx[:, i, j] = np.add(x[i], x[j]) / 2
                xx[:, j, i] = np.add(x[j], x[i]) / 2
    return xx

def diagonal_normalize(y, meanDist):
    """
    Removes diagonal effect from contact map y

    Inputs:
        y: contact map numpy array
        mean: mean contact frequency where mean[dist] is the mean at a given distance

    Outputs:
        result: new contact map
    """
    result = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(i + 1):
            distance = i - j
            exp_d = meanDist[distance]
            if exp_d == 0:
                # this is unlikely to happen
                pass
            else:
                result[i,j] = y[i,j] / exp_d
                result[j,i] = result[i,j]

    return result

def percentile_normalize(y, percentiles):
    """
    Performs percentile normalization on contact map y.

    The maximum value in y must at most percentiles[-1].

    Inputs:
        y: contact map numpy array
        percentiles: list of percentiles (frequencies)

    Outputs:
        result: new contact map
    """

    result = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(i + 1):
            val = y[i,j]
            p = 0
            while p < len(percentiles)-1 and percentiles[p] < val:
                p += 1
            result[i,j] = p
            result[j,i] = p

    return result

# plotting helper functions
def getFrequencies(dataFolder, diag, n, k, chi):
    # calculates number of times each interaction frequency
    # was observed
    converter = InteractionConverter(k)
    samples = make_dataset(dataFolder)
    freq_arr = np.zeros((int(n * (n+1) / 2 * len(samples)), 4)) # freq, sample, type, psi_ij
    ind = 0
    for sample in samples:

        sampleid = int(os.path.split(sample)[-1][6:])

        x = np.load(os.path.join(sample, 'x.npy'))
        if diag:
            y = np.load(os.path.join(sample, 'y_diag_norm.npy'))
        else:
            y = np.load(os.path.join(sample, 'y.npy'))
        for i in range(n):
            xi = x[i]
            for j in range(i+1):
                xj = x[j]
                comb = frozenset({tuple(xi), tuple(xj)})
                comb_type = converter.comb2Type(comb)
                psi_ij = xi @ chi @ xj
                freq_arr[ind] = [y[i,j], sampleid, comb_type, psi_ij]
                ind += 1

    return freq_arr

def getPercentiles(arr, prcnt_arr):
    """Helper function to get multiple percentiles at once."""
    result = np.zeros_like(prcnt_arr).astype(np.float64)
    arr_sort = np.sort(arr.flatten())
    for i, p in enumerate(prcnt_arr):
        result[i] = np.percentile(arr, p)
    return result

def generateDistStats(y, mode = 'freq', stat = 'mean'):
    '''
    Calculates statistics of contact frequency/probability as a function of distance

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

# Plotting functions
def plotFrequenciesSubplot(freq_arr, dataFolder, diag, k, sampleid, split = 'type', xmax = None):
    """
    Plotting function for frequency distributions corresponding to only one sample.

    Inputs:
        freq_arr: numpy array with 4 columns (freq, sampleid, interaction_type, psi)
        dataFolder: location of data, used for saving image
        diag: whether diagonal was normalized
        sampleid: int, which sample id to plot
        k: number of epigentic marks, used for InteractionConverter
        split: how to split data into subplots: None for no subplots, type for subplot on interaction type, psi for sublot on interation psi
        xmax: x axis limit
    """
    freq_pd = pd.DataFrame(freq_arr, columns = ['freq', 'sampleID', 'type', 'psi']) # cols = freq, sampleid, interaction_type
    converter = InteractionConverter(k)

    if split is None:
        fig = plt.figure(figsize=(10, 5))
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
                ax.hist(g_df['freq'], bins = 100)
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
                for g_name_2, g_df_2 in g_df.groupby(['psi']):
                    ax = fig.add_subplot(1, 3, indplt)
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
    bigax.set_xlabel('contact frequency', fontsize = 16)
    bigax.set_ylabel('count of contact frequency', fontsize = 16)

    fig.tight_layout()
    if diag:
        fig.suptitle('sample{} post normalization'.format(sampleid), fontsize = 16, y = 1)
    else:
        fig.suptitle('sample{} pre normalization'.format(sampleid), fontsize = 16, y = 1)

    if xmax is not None:
        plt.xlim(right = xmax)

    plt.savefig(os.path.join(dataFolder, 'samples', "sample{}".format(sampleid), 'freq_count_sample{}_diag_{}_split_{}.png'.format(sampleid, diag, split)))
    plt.close()

def plotFrequenciesSampleSubplot(freq_arr, dataFolder, diag, k, split = 'type'):
    """
    Plotting function for frequency distributions where each subplot corresponds to one sample.

    Inputs:
        freq_arr: numpy array with 4 columns (freq, sampleid, interaction_type, psi)
        dataFolder: location of data, used for saving image
        diag: whether diagonal was normalized
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
    if diag:
        fig.suptitle('post normalization', fontsize = 16, y = 1)
    else:
        fig.suptitle('pre normalization', fontsize = 16, y = 1)

    plt.savefig(os.path.join(dataFolder, 'freq_count_multisample_diag_{}_split_{}.png'.format(diag, split)))
    plt.close()

def plotDistStats(datafolder, diag, ofile, mode = 'freq', stat = 'mean'):
    """
    Function to plot expected interaction frequency as a function of distance for all samples in dataFolder.

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
            y = np.load(os.path.join(sample, 'y_diag_norm.npy'))
        else:
            y = np.load(os.path.join(sample, 'y.npy'))
        result = generateDistStats(y, mode = mode, stat = stat)
        ax.plot(result, label = os.path.split(sample)[1])
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
        plt.title('post normalization', fontsize = 16)
    else:
        plt.title('pre normalization', fontsize = 16)
    plt.legend()
    plt.savefig(ofile)
    plt.close()

def plotModelFromDir(dir, model, ofile):
    saveDict = torch.load(dir, map_location=torch.device('cpu'))
    model.load_state_dict(saveDict['model_state_dict'])
    train_loss_arr = saveDict['train_loss']
    plt.plot(np.arange(0, epochs), train_loss_arr, label = 'train loss')
    plt.xlabel('epoch', fontsize = 16)
    plt.ylabel('loss', fontsize = 16)
    plt.legend()
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def plotModelFromArrays(train_loss_arr, ofile, val_loss = None):
    plt.plot(np.arange(0, len(train_loss_arr)), train_loss_arr, label = 'train loss')
    if val_loss is not None:
        plt.axhline(y = val_loss, color = 'black', linestyle = '--', label = 'final val loss')
    plt.xlabel('epoch', fontsize = 16)
    plt.ylabel('loss', fontsize = 16)
    plt.legend()
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def plotContactMap(y, ofile, title = None, vmax = 1, size_in = 10, minVal = None, maxVal = None, prcnt = False):
    """
    Plotting function for contact maps.

    Inputs:
        y: contact map numpy array
        ofile: save location
        title: plot title
        vmax: maximum value for color bar, 'mean' to set as mean value
        size_in: size of figure x,y in inches
        minVal: values in y less than minVal are set to 0
        maxVal: values in y greater than maxVal are set to 0
    """
    if prcnt:
        mycmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0,       'white'),
                                                  (0.25,    'orange'),
                                                  (0.5,     'red'),
                                                  (0.74,    'purple'),
                                                  (1,       'blue')], N=10)
    else:
        mycmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0,    'white'),
                                                  (1,    'red')], N=126)
    if len(y.shape) > 2:
        N, C, H, W = y.shape
        assert N == 1 and C == 1
        y = y.reshape(H,W)

    if minVal is not None or maxVal is not None:
        y = y.copy() # prevent issues from reference type
    if minVal is not None:
        ind = y < minVal
        y[ind] = 0
    if maxVal is not None:
        ind = y > maxVal
        y[ind] = 0
    plt.figure(figsize = (size_in, size_in))
    if vmax == 'mean':
        vmax = np.mean(y)
    elif vmax == 'max':
        vmax = np.max(y)
    ax = sns.heatmap(y, linewidth=0, vmin = 0, vmax = vmax, cmap = mycmap)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()


def getBaseParser():
    """Helper function to get default command line argument parser."""
    parser = argparse.ArgumentParser(description='Base parser')

    #pre-processing args
    parser.add_argument('--y_norm', type=str, default=None)
    parser.add_argument('--crop', type=str, default=None, help='size of crop to apply to image - format: <leftcrop-rightcrop>')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5, help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2, help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning eate. Default=0.001')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--milestones', type=str, help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr decay')

    # model args
    parser.add_argument('--data_folder', type=str, default='dataset_04_18_21', help='Location of data')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--ifile_folder', type=str, default='models/', help='Location of input file for pretrained model')
    parser.add_argument('--ifile', type=str, help='Name of input file for pretrained model')
    parser.add_argument('--ofile_folder', type=str, default='models/', help='Location to save checkpoint models')
    parser.add_argument('--ofile', type=str, default='model', help='Name of save file')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')

    return parser

def roundUpBy10(val):
    """Rounds value up to the nearst multiple of 10."""
    assert val > 0, "val too small"
    assert val < 10**10, "val too big"
    mult = 1
    while val > mult:
        mult *= 10
    return mult

def str2bool(v):
    """
    Helper function for argparser, converts str to boolean for various string inputs.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Inputs:
        v: string
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class InteractionConverter():
    """Class that allows conversion between epigenetic mark bit string pairs and integer type id"""
    def __init__(self, k, chi = None):
        self.k = k
        self.chi = chi
        self.allStrings = self.generateAllBinaryStrings()
        self.comb2TypeDict = {}
        self.type2CombDict = {}

        curr_type = 0
        n = len(self.allStrings)
        for i in range(n):
            xi = self.allStrings[i]
            for j in range(n):
                xj = self.allStrings[j]
                comb = frozenset({tuple(xi), tuple(xj)})
                if comb not in self.comb2TypeDict.keys():
                    self.comb2TypeDict[comb] = curr_type
                    self.type2CombDict[curr_type] = comb
                    curr_type += 1

        self.types = np.arange(0, curr_type, 1)

        if self.chi is not None and self.x is not None:
            self.setPsi()

    def setChi(self, chi):
        self.chi = chi

    def setPsi(self):
        assert self.chi is not None, "set chi first"
        self.Psi = self.allStrings @ self.chi @ self.allStrings.T

    def getPsi_ij(self, xi, xj):
        assert self.chi is not None, "set chi first"
        return xi @ self.chi @ xj

    def comb2Type(self, comb):
        # input comb must be a frozenset
        return self.comb2TypeDict[comb]

    def type2Comb(self, type):
        # input type must be an integer
        return self.type2CombDict[type]

    def comb2str(self, comb):
        combList = sorted(list(comb))
        if len(combList) == 2:
            return '{} - {}'.format(combList[0], combList[1])
        else:
            return '{} - {}'.format(combList[0], combList[0])

    def fillArrayWithAllBinaryStrings(self, n, arr, temp_arr, i, row = 0):
        # https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/
        if i == n:
            arr.append(temp_arr.copy())
            row += 1
            return row

        # First assign "1" at ith position
        # and try for all other permutations
        # for remaining positions
        temp_arr[i] = 1
        self.fillArrayWithAllBinaryStrings(n, arr, temp_arr, i + 1, row)

        # And then assign "0" at ith position
        # and try for all other permutations
        # for remaining positions
        temp_arr[i] = 0
        self.fillArrayWithAllBinaryStrings(n, arr, temp_arr, i + 1, row)

    def generateAllBinaryStrings(self):
        arr = []
        temp_arr = [None]*self.k
        self.fillArrayWithAllBinaryStrings(self.k, arr, temp_arr, 0)
        np_arr = np.array(arr).astype(np.int8)
        return np_arr

def main():
    freq_counts_dict = getInteractionFrequencyCounts('dataset_04_08_21', diag = False)
    freq_arr = freqCountsDictToArray(freq_counts_dict)

if __name__ == '__main__':
    main()
