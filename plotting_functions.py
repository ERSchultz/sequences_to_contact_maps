import os
import os.path as osp
import sys
from shutil import rmtree

import torch
import torch.nn.functional as F

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pandas as pd
import time
from sklearn import metrics

from neural_net_utils.utils import *
from neural_net_utils.networks import *
from neural_net_utils.dataset_classes import *

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

    plt.savefig(osp.join(dataFolder, 'samples', "sample{}".format(sampleid), 'freq_count_sample{}_diag_{}_split_{}.png'.format(sampleid, diag, split)))
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

    plt.savefig(osp.join(dataFolder, 'freq_count_multisample_diag_{}_split_{}.png'.format(diag, split)))
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
            y = np.load(osp.join(sample, 'y_diag.npy'))
        else:
            y = np.load(osp.join(sample, 'y.npy'))
        result = generateDistStats(y, mode = mode, stat = stat)
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

def plotModelsFromDirs(dirs, imagePath, opts, log_y = False):
    # assume only difference in opts is lr
    # TODO good idea to check this with assert
    fig, ax = plt.subplots()
    colors = ['b', 'r', 'g', 'c']
    styles = ['-', '--']
    colors = colors[:len(dirs)]
    types = ['training', 'validation']
    lrs = []
    for dir, opt, c in zip(dirs, opts, colors):
        saveDict = torch.load(dir, map_location=torch.device('cpu'))
        train_loss_arr = saveDict['train_loss']
        val_loss_arr = saveDict['val_loss']
        if log_y:
            y_train = np.log10(train_loss_arr)
            y_val = np.log10(val_loss_arr)
        else:
            y_train = train_loss_arr
            y_val = val_loss_arr
        l1 = ax.plot(np.arange(1, len(train_loss_arr)+1), y_train, ls = styles[0], color = c)
        l2 = ax.plot(np.arange(1, len(val_loss_arr)+1), y_val, ls = styles[1], color = c)
        lrs.append(opt.lr)

    for c, lr in zip(colors, lrs):
        ax.plot(np.NaN, np.NaN, color = c, label = lr)

    ax2 = ax.twinx()
    for type, style in zip(types, styles):
        ax2.plot(np.NaN, np.NaN, ls = style, label = type, c = 'k')
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc = 1, title = 'lr')
    ax2.legend(loc = 3)

    ax.set_xlabel('Epoch', fontsize = 16)
    opt = opts[0]
    if opt.loss == 'mse':
        ylabel = 'MSE Loss'
    elif opt.loss == 'cross_entropy':
        ylabel = 'Cross Entropy Loss'
    elif opt.loss == 'BCE':
        ylabel = 'Binary Cross Entropy Loss'
    else:
        ylabel = 'Loss'
    if log_y:
        ylabel = r'$\log_{10}$(' + ylabel + ')'
    ax.set_ylabel(ylabel, fontsize = 16)

    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.y_norm is not None:
        y_norm = opt.y_norm.capitalize()
    else:
         y_norm = 'None'
    plt.title('Y Preprocessing: {}, Y Norm: {}'.format(preprocessing, y_norm), fontsize = 16)

    plt.tight_layout()
    if log_y:
        plt.savefig(osp.join(imagePath, 'train_val_loss_log.png'))
    else:
        plt.savefig(osp.join(imagePath, 'train_val_loss.png'))
    plt.close()

def plotModelFromDir(dir, imagePath, opt = None, log_y = False):
    """Wrapper function for plotModelFromArrays given saved model."""
    saveDict = torch.load(dir, map_location=torch.device('cpu'))
    train_loss_arr = saveDict['train_loss']
    val_loss_arr = saveDict['val_loss']
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt, log_y)

def plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt = None, log_y = False):
    """Plots loss as function of epoch."""
    if log_y:
        y_train = np.log10(train_loss_arr)
        y_val = np.log10(val_loss_arr)
    else:
        y_train = train_loss_arr
        y_val = val_loss_arr
    plt.plot(np.arange(1, len(train_loss_arr)+1), y_train, label = 'Training')
    plt.plot(np.arange(1, len(val_loss_arr)+1), y_val, label = 'Validation')

    ylabel = 'Loss'
    if opt is not None:
        if opt.loss == 'mse':
            ylabel = 'MSE Loss'
        elif opt.loss == 'cross_entropy':
            ylabel = 'Cross Entropy Loss'
        elif opt.loss == 'BCE':
            ylabel = 'Binary Cross Entropy Loss'

        if opt.y_preprocessing is not None:
            preprocessing = opt.y_preprocessing.capitalize()
        else:
            preprocessing = 'None'
        if opt.y_norm is not None:
            y_norm = opt.y_norm.capitalize()
        else:
             y_norm = 'None'
        upper_title = 'Y Preprocessing: {}, Y Norm: {}'.format(preprocessing, y_norm)
        lower_title = 'Final Validation Loss: {}'.format(np.round(val_loss_arr[-1], 3))
        plt.title('{}\n{}'.format(upper_title, lower_title), fontsize = 16)


        if opt.milestones is not None:
            lr = float(opt.lr)
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

    if log_y:
        ylabel = r'$\log_{10}$(' + ylabel + ')'
    plt.xlabel('Epoch', fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)

    plt.legend()
    plt.tight_layout()
    if log_y:
        plt.savefig(osp.join(imagePath, 'train_val_loss_log.png'))
    else:
        plt.savefig(osp.join(imagePath, 'train_val_loss.png'))
    plt.close()

def plotContactMap(y, ofile = None, title = None, vmin = 0, vmax = 1, size_in = 6, minVal = None, maxVal = None, prcnt = False, cmap = None):
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
    if cmap is None:
        if prcnt:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                     [(0,       'white'),
                                                      (0.25,    'orange'),
                                                      (0.5,     'red'),
                                                      (0.74,    'purple'),
                                                      (1,       'blue')], N=10)
        else:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
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
    ax = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap)
    if title is not None:
        plt.title(title, fontsize = 16)
    plt.tight_layout()
    if ofile is not None:
        plt.savefig(ofile)
    else:
        plt.show()
    plt.close()

def plotPerClassAccuracy(val_dataloader, imagePath, model, opt, title = None):
    """Plots accuracy for each class in percentile normalized contact map."""
    if opt.y_preprocessing == 'prcnt' and opt.loss == 'mse':
        return
        # when training on percentile preprocessed data using MSE loss it is impossible to convert back to classes

    acc_arr, freq_arr, acc_result = calculatePerClassAccuracy(val_dataloader, model, opt)

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


    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.y_norm is not None:
        y_norm = opt.y_norm.capitalize()
    else:
         y_norm = 'None'
    plt.title('Y Preprocessing: {}, Y Norm: {}\n{}'.format(preprocessing, y_norm, acc_result), fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(imagePath, 'per_class_acc.png'))

def plotDistanceStratifiedPearsonCorrelation(val_dataloader, imagePath, model, opt):
    """Plots Pearson correlation as a function of genomic distance"""
    p_arr = np.zeros((opt.valN, opt.n-1))
    P_arr_overall = np.zeros(opt.valN)
    model.eval()
    for i, data in enumerate(val_dataloader):
        data = data.to(opt.device)
        if opt.GNN_mode:
            if opt.autoencoder_mode:
                y = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr,
                                                        batch = data.batch,
                                                        max_num_nodes = opt.n)
            else:
                y = data.y
            yhat = model(data)
            minmax = data.minmax
            path = data.path[0]
        else:
            x, y, path, minmax = data
            path = path[0]
            yhat = model(x)
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        y = un_normalize(y, minmax)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt' and opt.loss == 'cross_entropy':
            yhat = np.argmax(yhat, axis = 1)
        else:
            yhat = un_normalize(yhat, minmax)
        yhat = yhat.reshape((opt.n,opt.n))

        overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
        if opt.verbose:
            print(overall_corr, corr_arr)
        P_arr_overall[i] = overall_corr
        p_arr[i, :] = corr_arr

    p_mean = np.mean(p_arr, axis = 0)
    np.save(osp.join(imagePath, 'distance_pearson_mean.npy'), p_mean)
    p_std = np.std(p_arr, axis = 0)
    np.save(osp.join(imagePath, 'distance_pearson_std.npy'), p_std)

    title = r'Overall Pearson R: {} $\pm$ {}'.format(np.round(np.mean(P_arr_overall), 3), np.round(np.std(P_arr_overall),3))
    print('Distance Stratified Pearson Correlation Results:', file = opt.log_file)
    print(title, end = '\n\n', file = opt.log_file)

    plt.plot(np.arange(opt.n-1), p_mean, color = 'black', label = 'mean')
    plt.fill_between(np.arange(opt.n-1), p_mean + p_std, p_mean - p_std, color = 'red', alpha = 0.5, label = 'std')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.legend(loc = 'lower left')

    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.y_norm is not None:
        y_norm = opt.y_norm.capitalize()
    else:
         y_norm = 'None'
    plt.title('Y Preprocessing: {}, Y Norm: {}\n{}'.format(preprocessing, y_norm, title), fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(imagePath, 'distance_pearson.png'))
    plt.close()

def plotPredictions(val_dataloader, model, opt, count = 5):
    print('Prediction Results:', file = opt.log_file)
    if opt.y_preprocessing != 'prcnt':
        prcntDist_path = osp.join(opt.data_folder, 'prcntDist.npy')
        prcntDist = np.load(prcntDist_path)
        print('prcntDist', prcntDist, file = opt.log_file)

    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.y_norm is not None:
        y_norm = opt.y_norm.capitalize()
    else:
         y_norm = 'None'
    upper_title = 'Y Preprocessing: {}, Y Norm: {}'.format(preprocessing, y_norm)


    loss_arr = np.zeros(opt.valN)
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        data = data.to(opt.device)
        if opt.GNN_mode:
            if opt.autoencoder_mode:
                y = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr,
                                                        batch = data.batch,
                                                        max_num_nodes = opt.n)
            else:
                y = data.y
            yhat = model(data)
            minmax = data.minmax
            path = data.path[0]
        else:
            x, y, path, minmax = data
            path = path[0]
            yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        y = un_normalize(y, minmax)
        yhat = yhat.cpu().detach().numpy()
        yhat = yhat.reshape((opt.n,opt.n))

        sample = osp.split(path)[-1]
        subpath = osp.join(opt.ofile_folder, sample)
        print(subpath, file = opt.log_file)
        if not osp.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        if opt.loss == 'mse':
            yhat_title = '{}\nY hat (MSE Loss: {})'.format(upper_title, np.round(loss, 3))
        elif opt.loss == 'cross_entropy':
            yhat_title = '{}\nY hat (Cross Entropy Loss: {})'.format(upper_title, np.round(loss, 3))
        elif opt.loss == 'BCE':
            yhat_title = '{}\nY hat (Binary Cross Entropy Loss: {})'.format(upper_title, np.round(loss, 3))
        else:
            yhat_title = '{}\nY hat (Loss: {})'.format(upper_title, np.round(loss, 3))

        loss_arr[i] = loss
        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        if opt.y_preprocessing == 'prcnt':
            plotContactMap(y, osp.join(subpath, 'y.png'), vmax = 'max', prcnt = True, title = 'Y')
            if opt.loss == 'cross_entropy':
                yhat = np.argmax(yhat, axis = 1)
                plotContactMap(yhat, osp.join(subpath, 'yhat.png'), vmax = 'max', prcnt = True, title = yhat_title)
            else:
                yhat = un_normalize(yhat, minmax)
                plotContactMap(yhat, osp.join(subpath, 'yhat.png'), vmax = 'max', prcnt = False, title = yhat_title)
        elif opt.y_preprocessing == 'diag' or opt.y_preprocessing == 'diag_instance':
            v_max = np.max(y)
            plotContactMap(y, osp.join(subpath, 'y.png'), vmax = v_max, prcnt = False, title = 'Y')
            yhat = un_normalize(yhat, minmax)
            plotContactMap(yhat, osp.join(subpath, 'yhat.png'), vmax = v_max, prcnt = False, title =yhat_title)

            # plot prcnt
            yhat_prcnt = percentile_preprocessing(yhat, prcntDist)
            plotContactMap(yhat_prcnt, osp.join(subpath, 'yhat_prcnt.png'), vmax = 'max', prcnt = True, title = 'Y hat prcnt')

            # plot dif
            ydif_abs = abs(yhat - y)
            plotContactMap(ydif_abs, osp.join(subpath, 'ydif_abs.png'), vmax = v_max, title = '|yhat - y|')
            ydif = yhat - y
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                     [(0, 'blue'),
                                                     (0.5, 'white'),
                                                      (1, 'red')], N=126)
            plotContactMap(ydif, osp.join(subpath, 'ydif.png'), vmin = -1 * v_max, vmax = v_max, title = 'yhat - y', cmap = cmap)
        else:
            raise Exception("Unsupported preprocessing: {}".format(opt.y_preprocessing))

    print('Loss: {} +- {}\n'.format(np.mean(loss_arr), np.std(loss_arr)), file = opt.log_file)

def freqDistributionPlots(dataFolder, n = 1024):
    chi = np.load(osp.join(dataFolder, 'chis.npy'))
    k = len(chi)

    # freq distribution plots
    for diag in [True, False]:
        print(diag)
        freq_arr = getFrequencies(dataFolder, diag, n, k, chi)
        for split in [None, 'type', 'psi']:
            print(split)
            plotFrequenciesSampleSubplot(freq_arr, dataFolder, diag, k, split)
            plotFrequenciesSubplot(freq_arr, dataFolder, diag, k, sampleid = 1, split = split)

def freqStatisticsPlots(dataFolder):
    # freq statistics plots
    for diag in [True, False]:
        for stat in ['mean', 'var']:
            ofile = osp.join(dataFolder, "freq_stat_{}_diag_{}.png".format(stat, diag))
            plotDistStats(dataFolder, diag, ofile, stat = stat)

def contactPlots(dataFolder):
    in_paths = sorted(make_dataset(dataFolder))
    for path in in_paths:
        print(path)
        y = np.load(osp.join(path, 'y.npy'))
        plotContactMap(y, osp.join(path, 'y.png'), title = 'pre normalization', vmax = 'mean')

        y_diag_norm = np.load(osp.join(path, 'y_diag.npy'))
        plotContactMap(y_diag_norm, osp.join(path, 'y_diag.png'), title = 'diag normalization', vmax = 'max')

        y_prcnt_norm = np.load(osp.join(path, 'y_prcnt.npy'))
        plotContactMap(y_prcnt_norm, osp.join(path, 'y_prcnt.png'), title = 'prcnt normalization', vmax = 'max', prcnt = True)

def plotROCCurve(val_dataloader, imagePath, model, opt):
    assert opt.GNN_mode and not opt.autoencoder_mode
    t0 = time.time()
    thresholds = np.linspace(0, 1, 51) # step size of 0.02
    tpr_array = np.zeros((51, opt.valN, opt.k))
    fpr_array = np.zeros((51, opt.valN, opt.k))
    acc_array = np.zeros((51, opt.valN))
    model.eval()
    for i, data in enumerate(val_dataloader):
        if opt.verbose:
            print('Iteration: {}'.format(i))
        data = data.to(opt.device)
        y = data.y
        y = y.cpu().numpy().astype(bool)
        y_not = np.logical_not(y)
        yhat = model(data)
        if opt.loss =='BCE':
            # usincg BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = F.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()
        minmax = data.minmax
        path = data.path[0]

        positives = np.sum(y, 0)
        negatives = np.sum(y_not, 0)
        for j,t in enumerate(thresholds):
            if opt.verbose:
                print('treshold: {}'.format(t))
            yhat_t = yhat > t
            tp = y & yhat_t
            tpr = np.sum(tp, 0) / positives
            fp = y_not & yhat_t
            fpr = np.sum(fp, 0) / negatives
            tpr_array[j, i] = tpr
            fpr_array[j, i] = fpr
            if opt.verbose:
                # print('yhat_t\n', yhat_t)
                print('tp: {}, p: {}, tpr: {}'.format(np.sum(tp, 0), positives, tpr))
                print('fp: {}, n: {}, fpr: {}\n'.format(np.sum(fp, 0), negatives, fpr))

            tn = y_not & np.logical_not(yhat_t)
            acc = np.sum(tp | tn) / opt.n / opt.k
            acc_array[j, i] = acc

    tpr_mean_array = np.round(np.mean(tpr_array, 1), 3)
    tpr_std_array = np.round(np.std(tpr_array, 1), 3)
    fpr_mean_array = np.round(np.mean(fpr_array, 1), 3)
    fpr_std_array = np.round(np.std(fpr_array, 1), 3)
    acc_mean_array = np.round(np.mean(acc_array, 1), 3)
    acc_std_array = np.round(np.std(acc_array, 1), 3)


    title = 'AUC:'
    for i in range(opt.k):
        area = np.round(metrics.auc(fpr_mean_array[:,i], tpr_mean_array[:,i]), 3)
        title += ' particle {} = {}'.format(i, area)
        plt.plot(fpr_mean_array[:,i], tpr_mean_array[:,i], label = 'particle {}'.format(i))
        plt.fill_between(fpr_mean_array[:,i],  tpr_mean_array[:,i] + tpr_std_array[:,i],  tpr_mean_array[:,i] - tpr_std_array[:,i], alpha = 0.5)
        plt.fill_between(fpr_mean_array[:,i],  tpr_mean_array[:,i] + tpr_std_array[:,i],  tpr_mean_array[:,i] - tpr_std_array[:,i], alpha = 0.5)
    plt.plot(np.linspace(0,1, 100), np.linspace(0,1,100), color = 'gray', linestyle='dashed')
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    plt.legend(loc = 'lower right')

    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.y_norm is not None:
        y_norm = opt.y_norm.capitalize()
    else:
         y_norm = 'None'
    plt.title('Y Preprocessing: {}, Y Norm: {}\n{}'.format(preprocessing, y_norm, title), fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(imagePath, 'ROC_curve.png'))
    plt.close()

    print('ROC Curve Results:', file = opt.log_file)
    print(title, end = '\n\n', file = opt.log_file)
    max_t = np.argmax(acc_mean_array)
    print('Max accuracy at t = {}: {} +- {}'.format(thresholds[max_t], acc_mean_array[max_t], acc_std_array[max_t]), file = opt.log_file)
    print('Corresponding FPR = {} +- {} and TPR = {} +- {}'.format(fpr_mean_array[max_t], fpr_std_array[max_t],
                                                                    tpr_mean_array[max_t], tpr_std_array[max_t]), file = opt.log_file)
    print('ROC time: {}'.format(np.round (time.time() - t0, 3)), file = opt.log_file)

def updateResultTables(model_type = None, mode = None):
    if model_type is None:
        model_types = ['Akita', 'DeepC', 'UNet', 'GNNAutoencoder']
        modes = [None, None, None, 'GNN']
    else:
        model_types = [model_type]
        modes = [mode]
    for model_type, mode in zip(model_types, modes):
        # set up header row
        opt_list = get_opt_header(model_type, mode)
        opt_list.extend(['Final Validation Loss', 'PCA Accuracy Mean', 'PCA Accuracy Std', 'PCA Spearman Mean', 'PCA Spearman Std', 'PCA Pearson Mean', 'PCA Pearson Std', 'Overall Pearson Mean', 'Overall Pearson Std'])
        results = [opt_list]

        # get data
        model_path = osp.join('results', model_type)
        parser = getBaseParser()
        for id in os.listdir(model_path):
            id_path = osp.join(model_path, id)
            if osp.isdir(id_path) and id.isdigit():
                txt_file = osp.join(id_path, 'argparse.txt')
                opt = parser.parse_args(['@{}'.format(txt_file)])
                opt.id = int(id)
                opt = finalizeOpt(opt, parser, True)
                opt_list = opt2list(opt)
                with open(osp.join(id_path, 'PCA_results.txt'), 'r') as f:
                    f.readline()
                    acc = f.readline().split(':')[1].strip().split(' +- ')
                    spearman = f.readline().split(':')[1].strip().split(' +- ')
                    pearson = f.readline().split(':')[1].strip().split(' +- ')
                with open(osp.join(id_path, 'out.log'), 'r') as f:
                    for line in f:
                        if line.startswith('Final val loss: '):
                            final_val_loss = line.split(':')[1].strip()
                        elif line.startswith('Overall Pearson R: '):
                            dist_pearson = line.split(':')[1].strip().split(' $\pm$ ')
                opt_list.extend([final_val_loss, acc[0], acc[1], spearman[0], spearman[1], pearson[0], pearson[1], dist_pearson[0], dist_pearson[1]])
                results.append(opt_list)

        ofile = osp.join(model_path, 'results_table.csv')
        with open(ofile, 'w', newline = '') as f:
            wr = csv.writer(f)
            wr.writerows(results)


def plotting_script(model, opt, train_loss_arr = None, val_loss_arr = None, dataset = None):
    if model is None:
        model = getModel(opt)
        model.to(opt.device)
        model_name = osp.join(opt.ofile_folder, 'model.pt')
        if osp.exists(model_name):
            save_dict = torch.load(model_name, map_location=torch.device('cpu'))
            model.load_state_dict(save_dict['model_state_dict'])
            train_loss_arr = save_dict['train_loss']
            val_loss_arr = save_dict['val_loss']
            print('Model is loaded: {}'.format(model_name), file = opt.log_file)
        else:
            raise Exception('Model does not exist: {}'.format(model_name))
        model.eval()

    opt.batch_size = 1 # batch size must be 1
    opt.shuffle = False # for reproducibility
    if dataset is None:
        dataset = getDataset(opt)
    _, val_dataloader, _ = getDataLoaders(dataset, opt)

    imagePath = opt.ofile_folder
    print('#### Plotting Script ####', file = opt.log_file)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt, True)

    if opt.plot:
        if opt.output_mode == 'contact':
            comparePCA(val_dataloader, imagePath, model, opt)

            plotDistanceStratifiedPearsonCorrelation(val_dataloader, imagePath, model, opt)

            plotPerClassAccuracy(val_dataloader, imagePath, model, opt)
        elif opt.output_mode == 'sequence':
            plotROCCurve(val_dataloader, imagePath, model, opt)
        else:
            raise Exception("Unkown output_mode {}".format(opt.output_mode))

    if opt.plot_predictions:
        assert opt.output_mode == 'contact', 'only contact predictions supported for plot_predictions not {}'.format(opt.output_mode)
        plotPredictions(val_dataloader, model, opt)

def updateAllPlots():
    parser = getBaseParser()
    for model_type in ['DeepC']: # 'Akita', 'UNet',
        print(model_type)
        model_path = osp.join('results', model_type)
        for id in os.listdir(model_path):
            print('\t', id)
            id_path = osp.join(model_path, id)
            if osp.isdir(id_path) and id.isdigit():
                txt_file = osp.join(id_path, 'argparse.txt')
                opt = parser.parse_args(['@{}'.format(txt_file)])
                opt.id = int(id)
                opt = finalizeOpt(opt, parser)

                # overwrite plotting flags
                opt.plot = False
                opt.plot_predictions = True
                print(opt, file = opt.log_file)
                plotting_script(None, opt)

def plotCombinedModels(modelType):
    path = osp.join('results', modelType)
    ids = [4, 5, 6]

    dirs = []
    opts = []
    parser = getBaseParser()
    for id in ids:
        id_path = osp.join(path, str(id))
        dirs.append(osp.join(id_path, 'model.pt'))
        txt_file = osp.join(id_path, 'argparse.txt')
        opt = parser.parse_args(['@{}'.format(txt_file)])
        opts.append(opt)
    imagePath = osp.join(path, '{} combined'.format(list2str(ids)))
    if not osp.exists(imagePath):
        os.mkdir(imagePath, mode = 0o755)

    for log in [True, False]:
        plotModelsFromDirs(dirs, imagePath, opts, log_y = log)

def main():
    opt = argparseSetup()
    plotting_script(None, opt)

    # cleanup
    if opt.root is not None and opt.delete_root:
        rmtree(opt.root)

if __name__ == '__main__':
    # plotCombinedModels('ContactGNN')
    # updateResultTables('GNNAutoencoder', 'GNN')
    # updateAllPlots()
    main()
    # freqDistributionPlots('dataset_04_18_21')
    # freqStatisticsPlots('dataset_04_18_21')
