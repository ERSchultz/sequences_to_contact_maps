import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.insert(0, dname)

import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pandas as pd
from utils import *
from networks import *
from utils import *
from plotting_functions import *
from dataset_classes import Sequences2Contacts

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

def plotModelFromDir(dir, imagePath, opt = None, log_y = False):
    """Wrapper function for plotModelFromArrays given saved model."""
    saveDict = torch.load(dir, map_location=torch.device('cpu'))
    train_loss_arr = saveDict['train_loss']
    val_loss_arr = saveDict['val_loss']
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt, log_y)

def plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt = None, log_y = False):
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
        plt.yscale('log', )
    plt.legend()
    plt.tight_layout()
    if log_y:
        plt.savefig(os.path.join(imagePath, 'train_val_loss_log.png'))
    else:
        plt.savefig(os.path.join(imagePath, 'train_val_loss.png'))
    plt.close()

def plotContactMap(y, ofile, title = None, vmax = 1, size_in = 6, minVal = None, maxVal = None, prcnt = False):
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

def plotPerClassAccuracy(val_dataloader, imagePath, model, opt, title = None):
    """Plots accuracy for each class in percentile normalized contact map."""
    if opt.y_preprocessing == 'prcnt' and opt.loss == 'mse':
        return
        # when training on percentile preprocessed data using MSE loss it is impossible to convert back to classes

    acc_arr, freq_arr = calculatePerClassAccuracy(val_dataloader, model, opt)

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
    plt.tight_layout()
    plt.savefig(os.path.join(imagePath, 'per_class_acc.png'))

def plotDistanceStratifiedPearsonCorrelation(val_dataloader, imagePath, model, opt):
    """Plots Pearson correlation as a function of genomic distance"""

    p_arr = np.zeros((opt.valN, opt.n-1))
    P_arr_overall = np.zeros(opt.valN)
    model.eval()
    for i, (x, y, path, minmax) in enumerate(val_dataloader):
        path = path[0]
        minmax = minmax
        assert x.shape[0] == 1, 'batch size must be 1 not {}'.format(x.shape[0])
        x = x.to(opt.device)
        y = y.to(opt.device)
        yhat = model(x)
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        y = un_normalize(y, minmax)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt' and opt.loss == 'cross_entropy':
            yhat = np.argmax(yhat, axis = 1)
        yhat = un_normalize(yhat, minmax)
        yhat = yhat.reshape((opt.n,opt.n))

        overall_corr, corr_arr = calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson')
        if opt.verbose:
            print(overall_corr, corr_arr)
        P_arr_overall[i] = overall_corr
        p_arr[i, :] = corr_arr

    p_mean = np.mean(p_arr, axis = 0)
    np.save(os.path.join(imagePath, 'distance_pearson_mean.npy'), p_mean)
    p_std = np.std(p_arr, axis = 0)
    np.save(os.path.join(imagePath, 'distance_pearson_std.npy'), p_std)

    title = r'Overall Pearson R: {} $\pm$ {}'.format(np.round(np.mean(P_arr_overall), 3), np.round(np.std(P_arr_overall),3))
    print('Distance Stratified Pearson Correlation Results:', file = opt.log_file)
    print(title, end = '\n\n', file = opt.log_file)

    plt.plot(np.arange(opt.n-1), p_mean, color = 'black', label = 'mean')
    plt.fill_between(np.arange(opt.n-1), p_mean + p_std, p_mean - p_std, color = 'red', alpha = 0.5, label = 'std')
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
    plt.legend(loc = 'lower left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(imagePath, 'distance_pearson.png'))
    plt.close()

def plotPredictions(val_dataloader, model, opt):
    print('Prediction Results:', file = opt.log_file)
    if opt.y_preprocessing != 'prcnt':
        prcntDist_path = os.path.join(opt.data_folder, 'prcntDist.npy')
        prcntDist = np.load(prcntDist_path)

    loss_arr = np.zeros(opt.valN)
    for i, (x, y, path, minmax) in enumerate(val_dataloader):
        assert x.shape[0] == 1, 'batch size must be 1 not {}'.format(x.shape[0])
        x = x.to(opt.device)
        y = y.to(opt.device)
        path = path[0]
        print(path, file = opt.log_file)
        if opt.mode == 'debugging':
            subpath = os.path.join(path, 'test')
        else:
            subpath = os.path.join(path, os.path.join(opt.model_type, str(opt.id)))
        if not os.path.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        if opt.loss == 'mse':
            yhat_title = 'Y hat (MSE Loss: {})'.format(np.round(loss, 3))
        elif opt.loss == 'cross_entropy':
            yhat_title = 'Y hat (Cross Entropy Loss: {})'.format(np.round(loss, 3))
        else:
            yhat_title = 'Y hat (Loss: {})'.format(np.round(loss, 3))
        loss_arr[i] = loss
        y = y.cpu().numpy()
        yhat = yhat.cpu().detach().numpy()
        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        y = un_normalize(y, minmax)
        if opt.y_preprocessing == 'prcnt':
            plotContactMap(y, os.path.join(subpath, 'y.png'), vmax = 'max', prcnt = True, title = 'Y')
            if opt.loss == 'cross_entropy':
                yhat = np.argmax(yhat, axis = 1)
                plotContactMap(yhat, os.path.join(subpath, 'yhat.png'), vmax = 'max', prcnt = True, title = yhat_title)
            else:
                yhat = un_normalize(yhat, minmax)
                plotContactMap(yhat, os.path.join(subpath, 'yhat.png'), vmax = 'max', prcnt = False, title = yhat_title)
        elif opt.y_preprocessing == 'diag':
            v_max = np.max(y)
            plotContactMap(y, os.path.join(subpath, 'y.png'), vmax = v_max, prcnt = False, title = 'Y')
            yhat = un_normalize(yhat, minmax)
            plotContactMap(yhat, os.path.join(subpath, 'yhat.png'), vmax = v_max, prcnt = False, title =yhat_title)

            # plot prcnt
            yhat_prcnt = percentile_preprocessing(yhat, prcntDist)
            plotContactMap(yhat_prcnt, os.path.join(subpath, 'yhat_prcnt.png'), vmax = 'max', prcnt = True, title = 'Y hat prcnt')

            # plot dif
            ydif = yhat - y
            plotContactMap(ydif, os.path.join(subpath, 'ydif.png'), vmax = v_max, title = 'difference')
        else:
            raise Exception("Unsupported preprocessing: {}".format(y_preprocessing))

    if opt.verbose:
        print('y', y, np.max(y))
        print('yhat', yhat, np.max(yhat))

    print('Loss: {} +- {}\n'.format(np.mean(loss_arr), np.std(loss_arr)), file = opt.log_file)

def freqDistributionPlots(dataFolder, n = 1024):
    chi = np.load(os.path.join(dataFolder, 'chis.npy'))
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
            ofile = os.path.join(dataFolder, "freq_stat_{}_diag_{}.png".format(stat, diag))
            plotDistStats(dataFolder, diag, ofile, stat = stat)

def contactPlots(dataFolder):
    in_paths = sorted(make_dataset(dataFolder))
    for path in in_paths:
        print(path)
        y = np.load(os.path.join(path, 'y.npy'))
        plotContactMap(y, os.path.join(path, 'y.png'), title = 'pre normalization', vmax = 'mean')

        y_diag_norm = np.load(os.path.join(path, 'y_diag.npy'))
        plotContactMap(y_diag_norm, os.path.join(path, 'y_diag.png'), title = 'diag normalization', vmax = 'max')

        y_prcnt_norm = np.load(os.path.join(path, 'y_prcnt.npy'))
        plotContactMap(y_prcnt_norm, os.path.join(path, 'y_prcnt.png'), title = 'prcnt normalization', vmax = 'max', prcnt = True)

def plotting_script(model, opt, train_loss_arr = None, val_loss_arr = None, load = False):
    opt.batch_size = 1 # batch size must be 1
    seq2ContactData = Sequences2Contacts(opt.data_folder, opt.toxx, opt.toxx_mode, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop, names = True, minmax = True)
    _, val_dataloader, _ = getDataLoaders(seq2ContactData, opt)
    if load:
        model_name = os.path.join(opt.ofile_folder, 'model.pt')
        if os.path.exists(model_name):
            save_dict = torch.load(model_name, map_location=torch.device('cpu'))
            model.load_state_dict(save_dict['model_state_dict'])
            train_loss_arr = save_dict['train_loss']
            val_loss_arr = save_dict['val_loss']
            print('Model is loaded: {}'.format(model_name), file = opt.log_file)
        else:
            raise Exception('Model does not exist: {}'.format(model_name))
    else:
        assert train_loss_arr is not None and val_loss_arr is not None



    imagePath = opt.ofile_folder
    print('#### Plotting Script ####', file = opt.log_file)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt, True)

    comparePCA(val_dataloader, imagePath, model, opt)

    plotDistanceStratifiedPearsonCorrelation(val_dataloader, imagePath, model, opt)

    plotPerClassAccuracy(val_dataloader, imagePath, model, opt)

    if opt.plot_predictions:
        plotPredictions(val_dataloader, model, opt)

def main():
    opt = argparseSetup()
    opt.mode = 'debugging'
    # overwrites if testing locally
    if opt.mode == 'debugging':
        opt.model_type = 'UNet'
        opt.data_folder = 'dataset_04_18_21'

        # Preprocessing
        opt.toxx = True
        opt.toxx_mode = 'add'
        opt.x_reshape = False

        # architecture
        opt.nf = 8
        opt.y_preprocessing = 'prcnt'
        opt.y_norm = None
        opt.loss = 'cross_entropy'
        opt.dilation_list = str2list('2-4-8-16-32-64-128-256-512')
        opt.kernel_w_list = str2list('5-5-5')
        opt.hidden_sizes_list = str2list('32-64-128')
        opt.dilation_list_trunk = str2list('2-4-8-16-32-64-128-256-512')
        opt.bottleneck = 32
        opt.dilation_list_head = str2list('2-4-8-16-32-64-128-256-512')
        # opt.out_act=nn.ReLU(True)
        opt.out_act = 'sigmoid'

        # hyperparameters
        opt.n_epochs = 15
        opt.lr = 1e-1
        opt.num_workers = 4
        opt.milestones = str2list('5-10')

        # other
        opt.plot_predictions = False
        opt.verbose = True

    if opt.model_type == 'UNet':
        opt.ofile = "UNet_nEpochs{}_nf{}_lr{}_milestones{}_yPreprocessing{}_yNorm{}".format(opt.n_epochs, opt.nf, float2str(opt.lr), list2str(opt.milestones), opt.y_preprocessing, opt.y_norm)
        if opt.loss == 'cross_entropy':
            assert opt.y_preprocessing == 'prcnt', 'must use percentile normalization with cross entropy'

            opt.criterion = F.cross_entropy

            # change default params
            opt.y_reshape = False
            opt.ydtype = torch.int64
            model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None) # activation combined into loss
        elif opt.loss == 'mse':
            opt.criterion = F.mse_loss
            model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
        else:
            raise Exception('Invalid loss: {}'.format(opt.loss))
    elif opt.model_type == 'DeepC':
        model = DeepC(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list, out_act = opt.out_act)
        opt.ofile = "DeepC_nEpochs{}_lr{}_milestones{}_yPreprocessing{}_yNorm{}_kernelW{}_hiddenSize{}_dilation{}".format(opt.n_epochs, float2str(opt.lr), list2str(opt.milestones), opt.y_preprocessing, opt.y_norm, list2str(opt.kernel_w_list), list2str(opt.hidden_sizes_list), list2str(opt.dilation_list))

        if opt.loss == 'mse':
            opt.criterion = F.mse_loss
        else:
            raise Exception('Invalid loss: {}'.format(opt.loss))
    elif opt.model_type == 'Akita':
        model = Akita(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list_trunk,
                            opt.bottleneck,
                            opt.dilation_list_head,
                            opt.out_act)
        opt.ofile = "Akita_nEpochs{}_lr{}_milestones{}_yPreprocessing{}_kernelW{}_hiddenSize{}_bottleneck{}".format(opt.n_epochs, float2str(opt.lr), list2str(opt.milestones), opt.y_preprocessing, list2str(opt.kernel_w_list), list2str(opt.hidden_sizes_list), opt.bottleneck)
        if opt.loss == 'mse':
            opt.criterion = F.mse_loss
        else:
            raise Exception('Invalid loss: {}'.format(opt.loss))
    else:
        print('Invalid model type: {}'.format(opt.model_type))
        # TODO

    print(opt, file = opt.log_file)
    model.to(opt.device)
    plotting_script(model, opt, load = True)

    # freqDistributionPlots('dataset_04_18_21')
    # freqStatisticsPlots('dataset_04_18_21')
    # contactPlots('dataset_04_18_21')


if __name__ == '__main__':
    main()
