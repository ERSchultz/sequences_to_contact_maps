import csv
import json
import math
import multiprocessing
import os
import os.path as osp
import sys

import imageio
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.decomposition import PCA
from sympy import solve, symbols

from .argparse_utils import get_base_parser, get_opt_header
from .InteractionConverter import InteractionConverter
from .load_utils import load_sc_contacts, load_X_psi
from .neural_net_utils import get_data_loaders, load_saved_model
from .utils import (calc_dist_strat_corr, calc_per_class_acc, compare_PCA,
                    triu_to_full, crop)
from .xyz_utils import (find_dist_between_centroids, find_label_centroid,
                        xyz_load, xyz_to_contact_grid, xyz_write)


#### Functions for plotting loss ####
def plot_combined_models(modelType, ids):
    path = osp.join('results', modelType)

    dirs = []
    opts = []
    parser = get_base_parser()
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

def plotModelsFromDirs(dirs, imagePath, opts, log_y = False):
    # check that only difference in opts is lr
    opt_header = get_opt_header(opts[0].model_type, opts[0].GNN_mode)
    opt_lists = []
    for opt in opts:
        opt_lists.append(opt2list(opt))

    for pos in range(len(opt_lists[0])):
        first = True
        for model in range(len(opt_lists)):
            if first:
                ref = opt_lists[model][pos]
                first = False
            elif opt_header[pos] != 'lr':
                assert opt_lists[model][pos] == ref

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
#### End section ####

def plot_seq_binary(seq, show = False, save = True, title = None, labels = None,
                    x_axis = True, ofile = 'seq.png'):
    '''Plotting function for *non* mutually exclusive binary particle types'''
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    plt.figure(figsize=(6, 3))
    for i, c in enumerate(colors):
        x = np.argwhere(seq[:, i] == 1)
        if labels is None:
            label_i = i
        else:
            label_i = labels[i]
        plt.scatter(x, np.ones_like(x) * i * 0.2, label = label_i, color = c['color'], s=3)

    plt.legend()
    ax = plt.gca()
    # ax.axes.get_yaxis().set_visible(False)
    if not x_axis:
        ax.axes.get_xaxis().set_visible(False)
    # else:
    #     ax.set_xticks(range(0, 1040, 40))
    #     ax.axes.set_xticklabels(labels = range(0, 1040, 40), rotation=-90)
    ax.set_yticks([i*0.2 for i in range(k)])
    ax.axes.set_yticklabels(labels = [f'mark {i}' for i in range(1,k+1)],
                            rotation='horizontal', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(ofile)
    if show:
        plt.show()
    plt.close()

def plotContactMap(y, ofile = None, title = None, vmin = 0, vmax = 1,
                    size_in = 6, minVal = None, maxVal = None, prcnt = False,
                    cmap = None, x_ticks = None, y_ticks = None):
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
    elif cmap == 'blue-red':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0, 'blue'),
                                                 (0.5, 'white'),
                                                  (1, 'red')], N=126)
    else:
        raise Exception('Invalid cmap: {}'.format(cmap))

    if len(y.shape) == 4:
        N, C, H, W = y.shape
        assert N == 1 and C == 1
        y = y.reshape(H,W)
    elif len(y.shape) == 3:
        N, H, W = y.shape
        assert N == 1
        y = y.reshape(H,W)
    else:
        H, W = y.shape

    if minVal is not None or maxVal is not None:
        y = y.copy() # prevent issues from reference type
    if minVal is not None:
        ind = y < minVal
        y[ind] = 0
    if maxVal is not None:
        ind = y > maxVal
        y[ind] = 0
    plt.figure(figsize = (size_in, size_in))

    # set min and max
    if vmin == 'min':
        vmin = np.percentile(y, 1)
        # uses 1st percentile instead of absolute min
    elif vmin == 'abs_min':
        vmin = np.min(y)
    if vmax == 'mean':
        vmax = np.mean(y)
    elif vmax == 'max':
        vmax = np.percentile(y, 99)
        # uses 99th percentile instead of absolute max
    elif vmax == 'abs_max':
        vmax = np.max(y)

    ax = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap)
    if x_ticks is None:
        pass
    elif len(x_ticks) == 0:
        ax.axes.get_xaxis().set_visible(False)
    else:
        ax.set_xticks([i+0.5 for i in range(W)])
        ax.axes.set_xticklabels(x_ticks)

    if y_ticks is None:
        pass
    elif len(y_ticks) == 0:
        ax.axes.get_yaxis().set_visible(False)
    else:
        ax.set_yticks([i+0.5 for i in range(H)])
        ax.axes.set_yticklabels(y_ticks, rotation='horizontal')

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
        # when training on percentile preprocessed data using MSE loss
        # it is impossible to convert back to classes

    acc_arr, freq_arr, acc_result = calc_per_class_acc(val_dataloader, model, opt)

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
    plt.title(f'Y Preprocessing: {preprocessing}, Y Norm: {y_norm}\n{acc_result}',
                fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(imagePath, 'per_class_acc.png'))
    plt.close()

def plotDistanceStratifiedPearsonCorrelation(val_dataloader, imagePath, model, opt):
    """Plots Pearson correlation as a function of genomic distance."""
    p_arr = np.zeros((opt.valN, opt.m-2))
    P_arr_overall = np.zeros(opt.valN)
    avg_arr = np.zeros(opt.valN)
    model.eval()
    for i, data in enumerate(val_dataloader):
        if opt.GNN_mode:
            data = data.to(opt.device)
            assert opt.autoencoder_mode
            y = data.contact_map
            y = torch.reshape(y, (-1, opt.m, opt.m))
            yhat = model(data)
            minmax = data.minmax
            path = data.path[0]
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        y = y.cpu().numpy().reshape((opt.m, opt.m))
        y = un_normalize(y, minmax)
        if opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt' and opt.loss == 'cross_entropy':
            yhat = np.argmax(yhat, axis = 1)
        else:
            yhat = un_normalize(yhat, minmax)
        yhat = yhat.reshape((opt.m,opt.m))

        triu_ind = np.triu_indices(opt.m)
        overall_corr, _ = stat(y[triu_ind], yhat[triu_ind])

        corr_arr, avg = calc_dist_strat_corr(y, yhat, mode = 'pearson', return_arr = True)
        avg = np.nanmean(corr_arr)
        if opt.verbose:
            print(overall_corr, corr_arr, avg)
        P_arr_overall[i] = overall_corr
        p_arr[i, :] = corr_arr
        avg_arr[i] = avg

    p_mean = np.mean(p_arr, axis = 0)
    np.save(osp.join(imagePath, 'distance_pearson_mean.npy'), p_mean)
    p_std = np.std(p_arr, axis = 0)
    np.save(osp.join(imagePath, 'distance_pearson_std.npy'), p_std)

    title = rf'''Overall Pearson R: {np.round(np.mean(P_arr_overall), 3)}
                $\pm$ {np.round(np.std(P_arr_overall),3)}'''
    title += rf'''\nAvg Dist Pearson R: {np.round(np.mean(avg_arr), 3)}
                $\pm$ {np.round(np.std(avg_arr),3)}'''
    print('Distance Stratified Pearson Correlation Results:', file = opt.log_file)
    print(title, end = '\n\n', file = opt.log_file)

    plt.plot(np.arange(opt.m-2), p_mean, color = 'black', label = 'mean')
    plt.fill_between(np.arange(opt.m-2), p_mean + p_std, p_mean - p_std,
                        color = 'red', alpha = 0.5, label = 'std')
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
    plt.title(f'Y Preprocessing: {preprocessing}, Y Norm: {y_norm}\n{title}',
                fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(imagePath, 'distance_pearson.png'))
    plt.close()

def plotEnergyPredictions(val_dataloader, model, opt, count = 5):
    print('Prediction Results:', file = opt.log_file)
    assert opt.output_mode == 'energy'
    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.y_norm is not None:
        y_norm = opt.y_norm.capitalize()
    else:
         y_norm = 'None'
    upper_title = 'Y Preprocessing: {}, Y Norm: {}'.format(preprocessing, y_norm)

    loss_arr = np.zeros(min(count, opt.valN))
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        if opt.GNN_mode:
            data = data.to(opt.device)
            y = data.energy
            y = torch.reshape(y, (-1, opt.m, opt.m))
            yhat = model(data)
            minmax = data.minmax
            path = data.path[0]
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        y = y.cpu().numpy().reshape((opt.m, opt.m))

        sample = osp.split(path)[-1]
        subpath = osp.join(opt.ofile_folder, sample)
        print('{}: {}'.format(subpath, loss), file = opt.log_file)
        if not osp.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        if opt.loss == 'mse':
            loss_title = 'MSE Loss'
        elif opt.loss == 'cross_entropy':
            loss_title = 'Cross Entropy Loss'
        elif opt.loss == 'BCE':
            loss_title = 'Binary Cross Entropy Loss'
        else:
            loss_title = 'Loss'

        if opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()
        yhat = yhat.reshape((opt.m,opt.m))

        yhat_title = '{}\n{} ({}: {})'.format(upper_title, r'$\hat{S}$',
                                                loss_title, np.round(loss, 3))

        loss_arr[i] = loss
        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        v_max = np.max(y)
        v_min = np.min(y)

        # not a contat map but using this plotContactMap function anyways
        plotContactMap(yhat, osp.join(subpath, 'energy_hat.png'), vmin = v_min,
                        vmax = v_max, cmap = 'blue-red', title = yhat_title)
        np.savetxt(osp.join(subpath, 'energy_hat.txt'), yhat, fmt = '%.3f')

        plotContactMap(y, osp.join(subpath, 'energy.png'), vmin = v_min,
                        vmax = v_max, cmap = 'blue-red', title = r'$S$')
        np.savetxt(osp.join(subpath, 'energy.txt'), y, fmt = '%.3f')

        # plot dif
        dif = yhat - y
        plotContactMap(dif, osp.join(subpath, 'edif.png'), vmin = -1 * v_max,
                        vmax = v_max, title = r'$\hat{S}$ - S', cmap = 'blue-red')

    print('Loss: {} +- {}\n'.format(np.mean(loss_arr), np.std(loss_arr)),
        file = opt.log_file)

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


    loss_arr = np.zeros(min(count, opt.valN))
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        if opt.GNN_mode:
            data = data.to(opt.device)
            assert opt.autoencoder_mode
            y = data.contact_map
            y = torch.reshape(y, (-1, opt.m, opt.m))
            yhat = model(data)
            minmax = data.minmax
            path = data.path[0]
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        y_torch = y # copy of torch version for PCA reconstruction
        y = y.cpu().numpy().reshape((opt.m, opt.m))

        sample = osp.split(path)[-1]
        subpath = osp.join(opt.ofile_folder, sample)
        print(subpath, file = opt.log_file)
        if not osp.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        if opt.loss == 'mse':
            loss_title = 'MSE Loss'
        elif opt.loss == 'cross_entropy':
            loss_title = 'Cross Entropy Loss'
        elif opt.loss == 'BCE':
            loss_title = 'Binary Cross Entropy Loss'
        else:
            loss_title = 'Loss'

        if opt.autoencoder_mode and opt.output_mode == 'contact':
            plotPCAReconstructions(y, y_torch, subpath, opt, loss_title, minmax)
        del y_torch

        y = un_normalize(y, minmax)
        if opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()
        yhat = yhat.reshape((opt.m,opt.m))

        yhat_title = '{}\nYhat ({}: {})'.format(upper_title, loss_title, np.round(loss, 3))

        loss_arr[i] = loss
        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        if opt.y_preprocessing == 'prcnt':
            plotContactMap(y, osp.join(subpath, 'y.png'), vmax = 'max', prcnt = True, title = 'Y')
            if opt.loss == 'cross_entropy':
                yhat = np.argmax(yhat, axis = 1)
                plotContactMap(yhat, osp.join(subpath, 'yhat.png'), vmax = 'max',
                                prcnt = True, title = yhat_title)
            else:
                yhat = un_normalize(yhat, minmax)
                plotContactMap(yhat, osp.join(subpath, 'yhat.png'), vmax = 'max',
                                prcnt = False, title = yhat_title)
        elif opt.y_preprocessing == 'diag' or opt.y_preprocessing == 'diag_instance':
            v_max = np.max(y)
            plotContactMap(y, osp.join(subpath, 'y.png'), vmax = v_max, prcnt = False, title = 'Y')
            yhat = un_normalize(yhat, minmax)
            plotContactMap(yhat, osp.join(subpath, 'yhat.png'), vmax = v_max,
                            prcnt = False, title = yhat_title)

            # plot prcnt
            yhat_prcnt = percentile_preprocessing(yhat, prcntDist)
            plotContactMap(yhat_prcnt, osp.join(subpath, 'yhat_prcnt.png'),
                            vmax = 'max', prcnt = True, title = r'$\hat{Y}$ prcnt')

            # plot dif
            ydif_abs = abs(yhat - y)
            plotContactMap(ydif_abs, osp.join(subpath, 'ydif_abs.png'),
                            vmax = v_max, title = r'|$\hat{Y}$ - Y|')
            ydif = yhat - y
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                     [(0, 'blue'),
                                                     (0.5, 'white'),
                                                      (1, 'red')], N=126)
            plotContactMap(ydif, osp.join(subpath, 'ydif.png'), vmin = -1 * v_max,
                            vmax = v_max, title = r'$\hat{Y}$ - Y', cmap = cmap)
        else:
            raise Exception("Unsupported preprocessing: {}".format(opt.y_preprocessing))

    print('Loss: {} +- {}\n'.format(np.mean(loss_arr), np.std(loss_arr)), file = opt.log_file)

def plotPCAReconstructions(y, y_torch, subpath, opt, loss_title, minmax):
    assert opt.autoencoder_mode and opt.output_mode == 'contact'
    for k in range(opt.k):
        pca = PCA(n_components = k)
        y_trans = pca.fit_transform(y)
        y_pca = pca.inverse_transform(y_trans)

        y_pca_torch = torch.tensor(y_pca, dtype = opt.ydtype)
        y_pca_torch = torch.reshape(y_pca_torch, (1, opt.m, opt.m))
        loss = opt.criterion(y_pca_torch, y_torch)

        y_pca = un_normalize(y_pca, minmax)
        plotContactMap(y_pca, osp.join(subpath, 'y_pc{}.png'.format(k+1)),
                    vmax = np.max(un_normalize(y, minmax)), prcnt = False,
                    title = 'Yhat Top {} PCs\n{}: {}'.format(k+1, loss_title,
                    np.round(loss, 3)))

def plotROCCurve(val_dataloader, imagePath, model, opt):
    assert opt.output_mode == 'sequence'
    # here y is the ground truth particle types
    # and yhat is the predicted particle types
    steps = 101 # step size of 0.01
    thresholds = np.linspace(0, 1, steps)
    tpr_array = np.zeros((steps, opt.valN, opt.k))
    fpr_array = np.zeros((steps, opt.valN, opt.k))
    acc_array = np.zeros((steps, opt.valN))
    model.eval()
    for i, data in enumerate(val_dataloader):
        if opt.verbose:
            print('Iteration: {}'.format(i))
        if opt.GNN_mode:
            data = data.to(opt.device)
            y = data.y
            yhat = model(data)
        elif opt.autoencoder_mode:
            x = data[0]
            x = x.to(opt.device)
            y = x
            yhat = model(x)
            y = torch.reshape(y, (opt.m, opt.k))
            yhat = torch.reshape(yhat, (opt.m, opt.k))
        y = y.cpu().numpy().astype(bool)
        y_not = np.logical_not(y)

        if opt.loss =='BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()

        if opt.verbose:
            print('y', y, y.shape)
            print('yhat', yhat, yhat.shape)

        positives = np.sum(y, 0)
        negatives = np.sum(y_not, 0)
        if opt.verbose:
            print('p: {}, n: {}'.format(positives, negatives))
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
            acc = np.sum(tp | tn) / opt.m / opt.k
            acc_array[j, i] = acc

    tpr_mean_array = np.round(np.mean(tpr_array, 1), 3)
    tpr_std_array = np.round(np.std(tpr_array, 1), 3)
    fpr_mean_array = np.round(np.mean(fpr_array, 1), 3)
    fpr_std_array = np.round(np.std(fpr_array, 1), 3)
    acc_mean_array = np.round(np.mean(acc_array, 1), 3)
    acc_std_array = np.round(np.std(acc_array, 1), 3)

    title = 'AUC: '
    for i in range(opt.k):
        area = np.round(metrics.auc(fpr_mean_array[:,i], tpr_mean_array[:,i]), 3)
        if i == 0:
            title += 'type {} = {}'.format(i, area)
        else:
             title += 'type {} = {}'.format(i, area)
        if i % 2 == 1:
            title += '\n'
        else:
            title += ' '
        plt.plot(fpr_mean_array[:,i], tpr_mean_array[:,i], label = 'particle type {}'.format(i))
        plt.fill_between(fpr_mean_array[:,i], tpr_mean_array[:,i] + tpr_std_array[:,i],
                        tpr_mean_array[:,i] - tpr_std_array[:,i], alpha = 0.5)
        plt.fill_between(fpr_mean_array[:,i], tpr_mean_array[:,i] + tpr_std_array[:,i],
                        tpr_mean_array[:,i] - tpr_std_array[:,i], alpha = 0.5)
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), color = 'gray', linestyle='dashed')
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
    plt.title(f'Y Preprocessing: {preprocessing}, Y Norm: {y_norm}\n{title.strip()}',
                fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(imagePath, 'ROC_curve.png'))
    plt.close()

    print('ROC Curve Results:', file = opt.log_file)
    print(title, end = '\n\n', file = opt.log_file)
    max_t = np.argmax(acc_mean_array)
    print(f'''Max accuracy at t = {thresholds[max_t]}:
            {acc_mean_array[max_t]} +- {acc_std_array[max_t]}''',
            file = opt.log_file)
    print(f'''Corresponding FPR = {fpr_mean_array[max_t]} +- {fpr_std_array[max_t]}
            and TPR = {tpr_mean_array[max_t]} +- {tpr_std_array[max_t]}''',
            file = opt.log_file)

def plot_top_PCs(inp, inp_type='', odir = None, log_file = sys.stdout, count = 2,
                plot = False, verbose = False):
    '''
    Plots top PCs of inp.

    Inputs:
        inp: np array containing input data
        inp_type: str representing type of input data
        odir: output directory to save plots to
        log_file: output file to write results to
        count: number of PCs to plot
        plot: True to plot
        verbose: True to print

    Outputs:
        pca.components_: all PCs of inp
    '''

    pca = PCA()
    try:
        pca = pca.fit(inp/np.std(inp, axis = 0))
    except ValueError:
        pca = pca.fit(inp)

    if verbose:
        if log_file is not None:
            print(f'\n{inp_type.upper()}', file = log_file)
            print(f'''% of total variance explained for first 4 PCs:
                {np.round(pca.explained_variance_ratio_[0:4], 3)}
                \n\tSum of first 4: {np.sum(pca.explained_variance_ratio_[0:4])}''',
                file = log_file)
            print(f'''Singular values for first 4 PCs:
                {np.round(pca.singular_values_[0:4], 3)}
                \n\tSum of all: {np.sum(pca.singular_values_)}''',
                file = log_file)

    if plot:
        i = 0
        while i < count:
            explained = pca.explained_variance_ratio_[i]
            if np.mean(pca.components_[i][:100]) < 0:
                PC = pca.components_[i] * -1
                # PCs are sign invariant, so this doesn't matter mathematically
                # goal is to help compare PCs visually by aligning them
            else:
                PC = pca.components_[i]
            plt.plot(PC)
            plt.title("Component {}: {}% of variance".format(i+1, np.round(explained * 100, 3)))
            if odir is not None:
                plt.savefig(osp.join(odir, '{}_PC_{}.png'.format(inp_type, i+1)))
                plt.close()
            else:
                plt.show()
            i += 1


    return pca.components_

#### Functions for plotting predicted particles ####
def plotParticleDistribution(val_dataloader, model, opt, count = 5, dims = (0,1),
                                use_latent = False):
    '''
    Plots distribution of particle type predictions for models that predict particle types.

    Here, x is ground truth particle type array; z is predicted particle type array.
    '''
    assert len(dims) == 2, 'currently only support 2D plots'
    converter = InteractionConverter(opt.k)
    all_binary_vectors = converter.generateAllBinaryStrings()
    model.eval()
    yhat = None
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        if opt.GNN_mode:
            data = data.to(opt.device)
            x = data.y # x is the simulated epigenetic marks
            minmax = data.minmax
            if opt.autoencoder_mode:
                assert opt.output_mode == 'contact'
                z = model.get_latent(data)
                yhat = model(data)
                yhat = yhat.cpu().detach().numpy().reshape((opt.m, opt.m))
                yhat = un_normalize(yhat, minmax)
            else:
                assert opt.output_mode == 'sequence'
                z = model(data)
                if opt.loss == 'BCE':
                    z = torch.sigmoid(z)
            minmax = data.minmax
            path = data.path[0]
        else:
            assert opt.output_mode == 'sequence'
            x, path = data
            path = path[0]
            x = x.to(opt.device)
            z = model(x)
            if opt.loss == 'BCE':
                z = torch.sigmoid(z)

        x = x.cpu().numpy().reshape((opt.m, opt.k))
        z = z.cpu().detach().numpy().reshape((opt.m, -1))

        sample = osp.split(path)[-1]
        subpath = osp.join(opt.ofile_folder, sample)
        print(subpath, file = opt.log_file)
        if not osp.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        np.save(osp.join(subpath,'z.npy'), z)

        # plots along polymer
        plotPredictedParticleTypesAlongPolymer(x, z, opt, subpath)
        if yhat is not None:
            plotPredictedParticlesVsPC(x, z, yhat, opt, subpath)


        # plots of distribution of predictions
        cmap = matplotlib.cm.get_cmap('tab10')
        ind = np.arange(len(all_binary_vectors)) % cmap.N
        colors = plt.cycler('color', cmap(ind))
        plt.rcParams["axes.prop_cycle"] = colors
        for vector in all_binary_vectors:
            ind = np.where((x == vector).all(axis = 1))
            plt.scatter(z[ind, dims[0]].reshape((-1)), z[ind, dims[1]].reshape((-1)),
                        label = vector)

        plt.xlabel('particle type {}'.format(dims[0]), fontsize = 16)
        plt.ylabel('particle type {}'.format(dims[1]), fontsize = 16)

        plt.ylim(0,1)
        plt.xlim(0,1)

        plt.legend(title = 'input particle type vector', title_fontsize = 16)
        plt.savefig(osp.join(subpath, f'particle_type_{dims[0]}_{dims[1]}_distribution_merged.png'))
        plt.close()

        # plot with subplots
        fig = plt.figure(figsize=(12, 12))
        bigax = fig.add_subplot(111, label = 'bigax')
        indplt = 1

        for vector, c in zip(all_binary_vectors, colors):
            ax = fig.add_subplot(opt.k, opt.k, indplt)
            ind = np.where((x == vector).all(axis = 1))
            ax.scatter(z[ind, dims[0]].reshape((-1)), z[ind, dims[1]].reshape((-1)),
                        color = c['color'])
            ax.set_title(f'particle type vector {vector}\n{len(ind[0])} particles',
                        fontsize = 16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            indplt += 1

        # Turn off axis lines and ticks of the big subplot
        bigax.spines['top'].set_color('none')
        bigax.spines['bottom'].set_color('none')
        bigax.spines['left'].set_color('none')
        bigax.spines['right'].set_color('none')
        bigax.tick_params(labelcolor = 'w', top = False, bottom = False, left = False, right = False)
        # set axis labels
        bigax.set_xlabel('particle type {}'.format(dims[0]), fontsize = 16)
        bigax.set_ylabel('particle type {}'.format(dims[1]), fontsize = 16)

        fig.tight_layout()

        plt.savefig(osp.join(subpath, f'particle_type_{dims[0]}_{dims[1]}_distribution.png'))
        plt.close()

def plotPredictedParticlesVsPC(x, z, yhat, opt, subpath):
    '''
    Plots the predicted particle types as a vector along the polymer
    compared to most similar principal component.
    '''
    cmap = matplotlib.cm.get_cmap('Set1')
    ind = np.arange(opt.k) % cmap.N
    colors = plt.cycler('color', cmap(ind))
    fig, ax = plt.subplots()

    styles = ['--', '-']
    types = ['predicted', 'PC']

    pca = PCA()
    pca.fit(yhat)

    # subplots
    fig = plt.figure(figsize=(12, 12))
    bigax = fig.add_subplot(111, label = 'bigax')
    indplt = 1

    rows = max(1, opt.k % 2)
    cols = math.ceil(opt.k / rows)

    for mark, c in enumerate(colors):
        ax = fig.add_subplot(rows, cols, indplt)
        ax.plot(np.arange(opt.m), z[:, mark], ls = styles[0], color = c['color'])
        max_corr = 0
        max_i = None
        for i in range(10):
            pc_i = pca.components_[i]
            corr, pval = pearsonr(pc_i, z[:, mark])
            corr = abs(corr)
            if corr > max_corr:
                max_i = i
                max_corr = corr
        ax.plot(np.arange(opt.m), pca.components_[max_i], ls = styles[1],
                color = c['color'], label = 'PC {}'.format(max_i))
        ax.set_title(f'particle type {mark}\nPearson R: {np.round(max_corr, 3)}', fontsize = 16)
        ax.legend()
        indplt += 1

    ax2 = bigax.twinx()
    for type, style in zip(types, styles):
        ax2.plot(np.NaN, np.NaN, ls = style, label = type, c = 'k')
    ax2.get_yaxis().set_visible(False)

    ax2.legend(loc = 3, title_fontsize = 16)

    # Turn off axis lines and ticks of the big subplot
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor = 'w', top = False, bottom = False, left = False, right = False)
    # set axis labels
    bigax.set_xlabel('particle index', fontsize = 16)
    bigax.set_ylabel('value', fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(subpath, 'particle_type_vector_predicted_PC.png'))
    plt.close()

def plotPredictedParticleTypesAlongPolymer(x, z, opt, subpath):
    '''Plots the predicted particle types as a vector along the polymer.'''
    # cycler: stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle
    cmap = matplotlib.cm.get_cmap('Set1')
    ind = np.arange(opt.k) % cmap.N
    colors = plt.cycler('color', cmap(ind))
    fig, ax = plt.subplots()

    styles = ['--', '-']
    types = ['predicted', 'true']

    # subplots
    fig = plt.figure(figsize=(12, 12))
    bigax = fig.add_subplot(111, label = 'bigax')
    indplt = 1

    rows = max(1, opt.k % 2)
    cols = math.ceil(opt.k / rows)

    for mark, c in enumerate(colors):
        ax = fig.add_subplot(rows, cols, indplt)
        ax.plot(np.arange(opt.m), z[:, mark], ls = styles[0], color = 'k')
        ax.plot(np.arange(opt.m), x[:, mark], ls = styles[1], color = c['color'])
        ax.set_title(f'particle type {mark}\n{np.sum(x[:, mark]).astype(int)} particles',
                        fontsize = 16)
        indplt += 1

    ax2 = bigax.twinx()
    for type, style in zip(types, styles):
        ax2.plot(np.NaN, np.NaN, ls = style, label = type, c = 'k')
    ax2.get_yaxis().set_visible(False)

    ax2.legend(loc = 3, title_fontsize = 16)

    # Turn off axis lines and ticks of the big subplot
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor = 'w', top = False, bottom = False, left = False, right = False)
    # set axis labels
    bigax.set_xlabel('particle index', fontsize = 16)
    bigax.set_ylabel('value', fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(subpath, 'particle_type_vector_predicted.png'))
    plt.close()
#### End section ####

#### Functions for plotting xyz files ####
def plot_config(xyz, L, x = None, ofile = None, show = True, title = None, legend=True):
    '''
    Plots particles in xyz as 3D scatter plot.
    Only supports mutually exclusive bead types for coloring. # TODO
    Inputs:
        xyz: shape (N,3) array of all particle positions
        L: side of LxLxL box (nm), if None the plot will be fit to the input
        x: bead types to color
        LJ: True if Lennard Jones particles
        ofile: location to save image
        show: if False fig, ax are returned
        title: title of plot
    '''
    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    # connect particles if polymer
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], color= '0.8')

    if x is None:
        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    else:
        # color unique types if x is not None
        types = np.argmax(x, axis = 1)
        n_types = np.max(types) + 1
        for t in range(n_types):
            condition = types == t
            # print(condition)
            ax.scatter(xyz[condition,0], xyz[condition,1], xyz[condition,2], label = t)
        if legend:
            plt.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if L is not None:
        #
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if ofile is not None:
        plt.savefig(ofile)
    if show:
        plt.show()
    plt.close()

def plot_xyz_gif():
    dir='/home/eric/dataset_test/samples/sample21'
    file = osp.join(dir, 'data_out/output.xyz')

    m=200

    x = np.load(osp.join(dir, 'x.npy'))[:m, :]
    xyz = xyz_load(file, multiple_timesteps=True)[:, :m, :]
    print(xyz.shape)
    xyz_write(xyz, osp.join(dir, 'data_out/output_x.xyz'), 'w', x = x)

    filenames = []
    for i in range(2, len(xyz)):
        fname=osp.join(dir, f'{i}.png')
        filenames.append(fname)
        plot_config(xyz[i, :, :], None, x = x, ofile = fname, show = False,
                    title = None, legend = False)

    # build gif
    # filenames = [osp.join(dir, f'ovito0{i}.png') for i in range(100, 900)]
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(osp.join(dir,'ovito.gif'), frames, format='GIF', fps=1)

    # remove files
    for filename in set(filenames):
        os.remove(filename)

def plot_sc_contact_maps(dataset, samples, ofolder = 'sc_contact', count = 20, jobs = 1, overall = True, N_max = None, crop_size = None):
    if isinstance(samples, int):
        samples = [samples]

    for sample in samples:
        print(f'sample{sample}')
        dir = osp.join(dataset, 'samples', f'sample{sample}')
        odir = osp.join(dir, ofolder)

        sc_contacts = load_sc_contacts(dir, zero_diag = True, jobs = jobs, triu = True, N_max = N_max)
        plot_sc_contact_maps_inner(sc_contacts, odir, count, jobs, overall, crop_size)

def plot_sc_contact_maps_inner(sc_contacts, odir, count, jobs, overall = False, crop_size = None):
    '''
    Plot sc contact map.

    Inputs:
        sc_contacts: np array of sc contact maps (in full or flattened upper traingle)
        odir: directory to write to
        count: number of plots to make
    '''
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o775)

    if len(sc_contacts.shape) == 3:
        triu = False
        N, m, _ = sc_contacts.shape
    else:
        triu = True
        N, l = sc_contacts.shape
        # infer m given length of upper triangle
        x, y = symbols('x y')
        y=x*(x+1)/2-l
        result=solve(y)
        m = int(np.max(result))

    overall_map = np.zeros((m,m))
    mapping = []
    for i in range(N):
        if overall or i % (N // count) == 0:
            if triu:
                # need to reconstruct from upper traingle
                contact_map = triu_to_full(sc_contacts[i, :], m)
            else:
                contact_map = sc_contacts[i, :, :]

            if crop_size is not None:
                contact_map = crop(contact_map, crop_size)

        if i % (N // count) == 0:
            np.savetxt(osp.join(odir, f'{i}.txt'), contact_map, fmt='%.2f')
            if jobs > 1:
                mapping.append((contact_map, osp.join(odir, f'{i}.png'), None, 0, 'abs_max'))
            else:
                plotContactMap(contact_map, osp.join(odir, f'{i}.png'), vmax = 'abs_max')

        if overall:
            overall_map += contact_map

    if jobs > 1:
        with multiprocessing.Pool(jobs) as p:
            p.starmap(plotContactMap, mapping)

    if overall:
        plotContactMap(overall_map, osp.join(odir, 'overall.png'), vmax = 'abs_max')


def plot_centroid_distance(dir = '/home/eric/dataset_test/samples',
                            samples = range(30, 36), parallel = False,
                            num_workers = None):
    if isinstance(samples, int):
        samples = [samples]
    if parallel:
        mapping = []
        for sample in samples:
            mapping.append((dir, sample))

        if num_workers is None:
            num_workers = len(samples)
        with multiprocessing.Pool(num_workers) as p:
            p.starmap(plot_centroid_distance_sample, mapping)
    else:
        for sample in samples:
            plot_centroid_distance_sample(dir, sample)

def plot_centroid_distance_sample(dir, sample):
    sample_dir = osp.join(dir, f'sample{sample}')
    xyz = xyz_load(osp.join(sample_dir, 'data_out', 'output.xyz'), multiple_timesteps = True)
    N, _, _ = xyz.shape
    _, psi = load_X_psi(osp.join(dir, f'sample{sample}'))
    m, k = psi.shape
    # TODO hard coded psi below
    k=3
    psi = np.zeros((m, 3))
    psi[:100, 0] = np.ones(100)
    psi[100:700, 1] = np.ones(600)
    psi[700:800, 2] = np.ones(100)

    distances = np.zeros((N, k, k))
    for i in range(N):
        centroids = find_label_centroid(xyz[i], psi)
        distances_i = find_dist_between_centroids(centroids)
        distances[i, :, :] = distances_i

    plt.hist(distances[:, 0, 2])
    plt.savefig(osp.join(sample_dir, 'AC_dist.png'))
    plt.close()

    plt.scatter(distances[:, 0, 2], np.linspace(0, N, N))
    plt.xlabel('A-C distance')
    plt.ylabel('sample index')
    plt.savefig(osp.join(sample_dir, 'AC_dist_vs_i.png'))
    plt.close()

### Primary scripts ###
def plotting_script(model, opt, train_loss_arr = None, val_loss_arr = None, dataset = None):
    if model is None:
        model, train_loss_arr, val_loss_arr = load_saved_model(opt, verbose = True)

    opt.batch_size = 1 # batch size must be 1
    opt.shuffle = False # for reproducibility
    if dataset is None:
        dataset = getDataset(opt, True, True)
    _, val_dataloader, _ = get_data_loaders(dataset, opt)

    imagePath = opt.ofile_folder
    print('#### Plotting Script ####', file = opt.log_file)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt, True)

    if opt.plot:
        if opt.output_mode == 'contact':
            compare_PCA(val_dataloader, imagePath, model, opt)
            plotDistanceStratifiedPearsonCorrelation(val_dataloader, imagePath, model, opt)
            plotPerClassAccuracy(val_dataloader, imagePath, model, opt)
        elif opt.output_mode == 'sequence':
            plotROCCurve(val_dataloader, imagePath, model, opt)
        elif opt.output_mode == 'energy':
            pass
            # TODO
        else:
            raise Exception("Unkown output_mode {}".format(opt.output_mode))


    if opt.plot_predictions:
        if opt.output_mode == 'contact':
            plotPredictions(val_dataloader, model, opt)
        elif opt.output_mode == 'sequence':
            if opt.model_type in {'ContactGNN', 'SequenceFCAutoencoder'}:
                plotParticleDistribution(val_dataloader, model, opt, use_latent = False)
            elif opt.model_type == 'GNNAutoencoder':
                plotParticleDistribution(val_dataloader, model, opt, use_latent = True)
        elif opt.output_mode == 'energy':
            plotEnergyPredictions(val_dataloader, model, opt)
