import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from numba import jit, njit
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import argparse
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
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

def getDataLoaders(dataset, opt):
    train_dataset, val_dataset, test_dataset = splitDataset(dataset, opt)
    train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size,
                                    shuffle = opt.shuffle, num_workers = opt.num_workers)
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        val_dataloader = None
    if len(val_dataset) > 0:
        test_dataloader = DataLoader(test_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader

def splitDataset(dataset, opt):
    """Splits input dataset into proportions specified by split."""
    opt.N = len(dataset)
    assert sum(opt.split) - 1 < 1e-5, "split doesn't sum to 1: {}".format(opt.split)
    opt.trainN = math.floor(opt.N * opt.split[0])
    opt.valN = math.floor(opt.N * opt.split[1])
    opt.testN = opt.N - opt.trainN - opt.valN
    return torch.utils.data.random_split(dataset,
                                        [opt.trainN, opt.valN, opt.testN],
                                        generator = torch.Generator().manual_seed(opt.seed))

# data processing functions
@njit
def x2xx(x, mode = 'mean'):
    # TODO better explanation here
    """
    Function for converting x to an image xx.

    For add:
        xx[k, i, j] = x[i, :] + x[j, :] (element-wise)
    For mean:
        xx[k, i, j] = (x[i, :] + x[j, :]) / 2 (element-wise)
    For append:
        xx[2k, i, j] = [x[i, :] ... x[j, :]]


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

def diagonal_preprocessing(y, meanDist):
    """
    Removes diagonal effect from contact map y.

    Inputs:
        y: contact map numpy array
        mean: mean contact frequency where mean[dist] is the mean at a given distance
        max: maximum value of

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

def percentile_preprocessing(y, percentiles):
    """
    Performs percentile preprocessing on contact map y.

    The maximum value in y must at most percentiles[-1].

    Inputs:
        y: contact map numpy array (2 dimensional)
        percentiles: list of percentiles (frequencies)

    Outputs:
        result: new contact map
    """

    if len(y.shape) > 2:
        N, C, H, W = y.shape
        assert N == 1 and C == 1
        y = y.reshape(H,W)

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
            y = np.load(os.path.join(sample, 'y_diag.npy'))
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
        diag: whether diagonal preprocessing was performed
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
        fig.suptitle('sample{} diag preprocessing'.format(sampleid), fontsize = 16, y = 1)
    else:
        fig.suptitle('sample{}'.format(sampleid), fontsize = 16, y = 1)

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
        diag: whether diagonal preprocessing was performed
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
        fig.suptitle('diag preprocessing', fontsize = 16, y = 1)
    else:
        fig.suptitle('no preprocessing', fontsize = 16, y = 1)

    plt.savefig(os.path.join(dataFolder, 'freq_count_multisample_diag_{}_split_{}.png'.format(diag, split)))
    plt.close()

def Stats(datafolder, diag, ofile, mode = 'freq', stat = 'mean'):
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
            y = np.load(os.path.join(sample, 'y_diag.npy'))
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
        plt.title('diag preprocessing', fontsize = 16)
    else:
        plt.title('no preprocessing', fontsize = 16)
    plt.legend()
    plt.savefig(ofile)
    plt.close()

def plotModelFromDir(dir, ofile, opt = None):
    """Wrapper function for plotModelFromArrays given saved model."""
    saveDict = torch.load(dir, map_location=torch.device('cpu'))
    train_loss_arr = saveDict['train_loss']
    val_loss_arr = saveDict['val_loss']
    plotModelFromArrays(train_loss_arr, val_loss_arr, ofile, opt)

def plotModelFromArrays(train_loss_arr, val_loss_arr, ofile, opt = None):
    """Plots loss as function of epoch."""
    plt.plot(np.arange(1, len(train_loss_arr)+1), train_loss_arr, label = 'Training')
    plt.plot(np.arange(1, len(val_loss_arr)+1), val_loss_arr, label = 'Validation')
    plt.xlabel('Epoch', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)

    if opt is not None:
        if opt.criterion == F.mse_loss:
            plt.ylabel('MSE Loss', fontsize = 16)
        elif opt.criterion == F.cross_entropy:
            plt.ylabel('Cross Entropy Loss', fontsize = 16)

        if opt.y_preprocessing is not None:
            preprocessing = opt.y_preprocessing.capitalize()
        else:
            preprocessing = 'None'
        plt.title('Y Preprocessing: {}'.format(preprocessing), fontsize = 16)

        if opt.milestones is not None:
            lr = opt.lr
            max_y = np.max(np.maximum(train_loss_arr, val_loss_arr))
            min_y = np.min(np.minimum(train_loss_arr, val_loss_arr))
            new_max_y = max_y + (max_y - min_y) * 0.1
            annotate_y = max_y + (max_y - min_y) * 0.05
            x_offset = (opt.milestones[0] - 1) * 0.05
            plt.ylim(top = new_max_y)
            plt.axvline(1, linestyle = 'dashed', color = 'green')
            plt.annotate('lr: {}'.format(lr), (1 + x_offset, annotate_y))
            for m in opt.milestones:
                lr = lr * opt.gamma
                plt.axvline(m, linestyle = 'dashed', color = 'green')
                plt.annotate('lr: {:.1e}'.format(lr), (m + x_offset, annotate_y))
    plt.legend()
    plt.tight_layout()
    plt.savefig(ofile)
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
    if len(y.shape) == 4:
        N, C, H, W = y.shape
        assert N == 1 and C == 1
        y = y.reshape(H,W)
    elif len(y.shape) == 3:
        N, H, W = y.shape
        assert N == 1
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

def plotPerClassAccuracy(acc_arr, freq_arr, title = None, ofile = None):
    N, C = acc_arr.shape
    width = 0.35
    x = np.arange(C)
    acc_arr_mu = np.mean(acc_arr, axis = 0)
    acc_arr_std = np.std(acc_arr, axis = 0)
    freq_arr_mu = np.mean(freq_arr, axis = 0)
    freq_arr_std = np.std(freq_arr, axis = 0)

    fig, ax1 = plt.subplots()
    color = 'tab:red'

    ax1.bar(x, acc_arr_mu, color = color, width = width, yerr = acc_arr_std)
    ax1.set_xlabel("Class", fontsize = 16)
    ax1.set_ylabel("Accuracy", fontsize = 16, color = color)
    ax1.tick_params(axis = 'y', labelcolor = color)
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(x)


    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.bar(x + width, freq_arr_mu, color = color, width = width, yerr = freq_arr_std)
    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.set_ylabel("Class Frequency", fontsize = 16, color = color)

    if title is not None:
        plt.title(title)
    if ofile is not None:
        plt.savefig(ofile)
    plt.show()

def plotDistanceStratifiedPearsonCorrelation(val_dataloader, model, ofile, opt):
    """Plots Pearson correlation as a function of genomic distance"""

    p_arr = np.zeros((opt.valN, opt.n-1))
    P_arr_overall = np.zeros(opt.valN)
    triu_ind = np.triu_indices(opt.n)
    model.eval()
    i = 0
    for x, y in val_dataloader:
        x = x.to(opt.device)
        y = y.to(opt.device)
        for j in range(y.shape[0]):
            # manually using batchsize of 1
            xj = x[j].unsqueeze(0)
            yj = y[j].unsqueeze(0)
            yhat = model(xj)
            yj = yj.cpu().numpy().reshape((opt.n, opt.n))
            yhat = yhat.cpu().detach().numpy()

            if opt.y_preprocessing == 'prcnt':
                yhat = np.argmax(yhat, axis = 1)
            yhat = yhat.reshape((opt.n,opt.n))


            corr, pval = pearsonr(yj[triu_ind], yhat[triu_ind])
            P_arr_overall[i] = corr

            for d in range(opt.n-1):
                y_diag = np.diagonal(yj, offset = d)
                yhat_diag = np.diagonal(yhat, offset = d)
                corr, pval = pearsonr(y_diag, yhat_diag)
                p_arr[i, d] = corr

    p_mean = np.mean(p_arr, axis = 0)
    p_std = np.std(p_arr, axis = 0)

    print('Pearson R: {} +- {}'.format(np.mean(P_arr_overall), np.std(P_arr_overall)))

    plt.plot(np.arange(opt.n-1), p_mean, color = 'black')
    plt.fill_between(np.arange(opt.n-1), p_mean + p_std, p_mean - p_std, color = 'red', alpha = 0.5)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

# other functions
def comparePCA(val_dataloader, model, opt):
    """Computes statistics of 1st PC of contact map"""
    acc_arr = np.zeros(opt.valN)
    rho_arr = np.zeros(opt.valN)
    p_arr = np.zeros(opt.valN)
    pca = PCA()
    model.eval()
    i = 0
    for x, y in val_dataloader:
        x = x.to(opt.device)
        print(x.shape)
        y = y.to(opt.device)
        print(y.shape)
        for j in range(y.shape[0]):
            print(j)
            # manually using batchsize of 1
            x = x[j].unsqueeze(0)
            y = y[j].unsqueeze(0)
            yhat = model(x)
            y = y.cpu().numpy().reshape((opt.n, opt.n))
            yhat = yhat.cpu().detach().numpy()

            if opt.y_preprocessing == 'prcnt':
                yhat = np.argmax(yhat, axis = 1)
            yhat = yhat.reshape((opt.n, opt.n))

            result_y = pca.fit(y)
            comp1_y = pca.components_[0]
            sign1_y = np.sign(comp1_y)

            result_yhat = pca.fit(yhat)
            comp1_yhat = pca.components_[0]
            sign1_yhat = np.sign(comp1_yhat)
            acc = np.sum((sign1_yhat == sign1_y)) / sign1_y.size
            acc_arr[i] = acc

            corr, pval = spearmanr(comp1_yhat, comp1_y)
            rho_arr[i] = corr

            corr, pval = pearsonr(comp1_yhat, comp1_y)
            p_arr[i] = corr
            i += 1

    print('PCA results:')
    print(p_arr)
    print('Accuracy: {} +- {}'.format(np.mean(acc_arr), np.std(acc_arr)))
    print('Spearman R: {} +- {}'.format(np.mean(rho_arr), np.std(rho_arr)))
    print('Pearson R: {} +- {}'.format(np.mean(p_arr), np.std(p_arr)))
    print()

def argparseSetup():
    """Helper function to get default command line argument parser."""
    parser = argparse.ArgumentParser(description='Base parser')

    # pre-processing args
    parser.add_argument('--y_preprocessing', type=str, default='diag', help='type of pre-processing for y')
    parser.add_argument('--classes', type=int, default=10, help='number of classes in percentile normalization')
    parser.add_argument('--y_norm', type=str, default='batch', help='type of [0,1] normalization for y')
    parser.add_argument('--crop', type=str2list, help='size of crop to apply to image - format: <leftcrop-rightcrop>')
    parser.add_argument('--toxx', type=str2bool, default=False, help='True if x should be converted to 2D image')
    parser.add_argument('--x_reshape', type=str2bool, default=True, help='True if x should be considered a 1D image')

    # dataloader args
    parser.add_argument('--split', type=str2list, default=[0.8, 0.1, 0.1], help='Train, val, test split for dataset')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Whether or not to shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5, help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2, help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning eate. Default=0.001')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--milestones', type=str2list, default = [2], help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr decay')
    parser.add_argument('--loss', type=str, default='mse', help='Type of loss to use: options: {"mse", "cross_entropy"}')

    # model args
    parser.add_argument('--model_type', type=str, help='Type of model')
    parser.add_argument('--data_folder', type=str, default='test', help='Location of data')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='True if using a pretrained model')
    parser.add_argument('--resume_training', type=str2bool, default=False, help='True if resuming traning of a partially trained model')
    parser.add_argument('--ifile_folder', type=str, default='models/', help='Location of input file for pretrained model')
    parser.add_argument('--ifile', type=str, help='Name of input file for pretrained model')
    parser.add_argument('--ofile_folder', type=str, default='models/', help='Location to save checkpoint models')
    parser.add_argument('--ofile', type=str, default='model', help='Name of save file')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')

    # post-processing args
    parser.add_argument('--plot', type=str2bool, default=False, help='True to plot predictions') # TODO use this

    # SimpleEpiNet args
    parser.add_argument('--kernel_w_list', type=str2list, default=[5,5], help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=str2list, default=[10,10], help='List of hidden sizes for convolutional layers')

    # UNet args
    parser.add_argument('--nf', type=int, default=8, help='Number of filters')

    # DeepC args
    parser.add_argument('--dilation_list', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers')
    parser.add_argument('--hidden_size_dilation', type=int, default=10, help='Hidden size of dilated convolutional layers')

    # Akita args
    parser.add_argument('--dilation_list_trunk', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers of trunk')
    parser.add_argument('--bottleneck', type=int, default=10, help='Number of filters in bottleneck (must be <= hidden_size_dilation_trunk)')
    parser.add_argument('--dilation_list_head', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers of head')


    opt = parser.parse_args()
    # configure cuda
    if opt.gpus > 1:
        opt.cuda = True
        opt.use_parallel = True
        opt.gpu_ids = []
        for ii in range(6):
            try:
                torch.cuda.get_device_properties(ii)
                print(str(ii))
                opt.gpu_ids.append(ii)
            except AssertionError:
                print('Not ' + str(ii) + "!")
    elif opt.gpus == 1:
        opt.cuda = True
        opt.use_parallel = False
    else:
        opt.cuda = False
        opt.use_parallel = False

    if opt.cuda and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        opt.cuda = False
        opt.use_parallel = False

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


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

def str2list(v):
    """
    Helper function for argparser, converts str to list by splitting on '-': "i-j-k" -> [i,j,k].

    Inputs:
        v: string
    """
    if v is None:
        return None
    if isinstance(v, str):
       return [int(i) for i in v.split('-')]
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def list2str(v):
    """
    Helper function to undo str2list.

    Inputs:
        v: list
    """
    assert type(v) == list
    return '-'.join([str(i) for i in v])


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
    parser = getBaseParser()
    opt = parser.parse_args()
    print(opt)
    # plotPerClassAccuracy(None, None, 5)

if __name__ == '__main__':
    main()
