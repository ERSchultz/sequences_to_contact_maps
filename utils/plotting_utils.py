import copy
import json
import math
import multiprocessing
import os
import os.path as osp
import sys
import tarfile
from shutil import rmtree

import imageio
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sympy import solve, symbols

from utils.energy_utils import calculate_diag_chi_step

from .argparse_utils import (ArgparserConverter, finalize_opt, get_base_parser,
                             get_opt_header, opt2list)
from .InteractionConverter import InteractionConverter
from .load_utils import load_sc_contacts, load_X_psi
from .neural_net_utils import get_data_loaders, get_dataset, load_saved_model
from .utils import DiagonalPreprocessing, crop, triu_to_full
from .xyz_utils import (find_dist_between_centroids, find_label_centroid,
                        xyz_load)


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
        opt = finalize_opt(opt, parser, local = True, debug = True)
        opts.append(opt)

    imagePath = osp.join(path, '{} combined'.format(ArgparserConverter.list2str(ids)))
    if not osp.exists(imagePath):
        os.mkdir(imagePath, mode = 0o755)

    for log in [True, False]:
        plotModelsFromDirs(dirs, imagePath, opts, log_y = log)

def plotModelsFromDirs(dirs, imagePath, opts, log_y = False):
    # check that only one param is different
    opt_header = get_opt_header(opts[0].model_type, opts[0].GNN_mode)
    opt_lists = []
    for opt in opts:
        opt_lists.append(opt2list(opt))

    differences_names = []
    differences = []
    ids = []
    for pos in range(len(opt_lists[0])):
        first = True
        for model in range(len(opt_lists)):
            if first:
                ref = opt_lists[model][pos]
                first = False
            else:
                if opt_lists[model][pos] != ref:
                    param = opt_header[pos]
                    if param not in {'id', 'resume_training'}:
                        differences_names.append(param)
                        differences.append((ref, opt_lists[model][pos]))
                    if param == 'id':
                        ids = (ref, opt_lists[model][pos])


    if len(differences) == 1:
        diff_name = differences_names.pop()
        diff = differences.pop()
        if diff_name == 'edge_transforms':
            new_diff = ([], [])
            for a, b in zip(diff[0], diff[1]):
                if a != b:
                    new_diff[0].append(a)
                    new_diff[1].append(b)
            diff = new_diff


    else:
        print('Differences:')
        for name, (a, b) in zip(differences_names, differences):
            print(f'{name}:\n\t{a}\n\t{b}')
        diff_name = 'id'
        diff = ids
        print(diff)

    fig, ax = plt.subplots()
    colors = ['b', 'r', 'g', 'c']
    styles = ['-', '--']
    colors = colors[:len(dirs)]
    types = ['training', 'validation']
    labels = []
    for dir, opt, c in zip(dirs, opts, colors):
        saveDict = torch.load(dir, map_location=torch.device('cpu'))
        train_loss_arr = saveDict['train_loss']
        val_loss_arr = saveDict['val_loss']
        l1 = ax.plot(np.arange(1, len(train_loss_arr)+1), train_loss_arr,
                    ls = styles[0], color = c)
        if log_y:
            ax.set_yscale('log')
        l2 = ax.plot(np.arange(1, len(val_loss_arr)+1), val_loss_arr,
                    ls = styles[1], color = c)

    for c, label_i in zip(colors, diff):
        ax.plot(np.NaN, np.NaN, color = c, label = label_i)

    ax2 = ax.twinx()
    for type, style in zip(types, styles):
        ax2.plot(np.NaN, np.NaN, ls = style, label = type, c = 'k')
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc = 1, title = diff_name)
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
        ylabel = f'Semi-log {ylabel}'
    ax.set_ylabel(ylabel, fontsize = 16)

    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.preprocessing_norm is not None:
        preprocessing_norm = opt.preprocessing_norm.capitalize()
    else:
         preprocessing_norm = 'None'
    plt.title(f'Y Preprocessing: {preprocessing}, Norm: {preprocessing_norm}',
                fontsize = 16)

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

def plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt = None,
                        log_y = False):
    """Plots loss as function of epoch."""
    plt.plot(np.arange(1, len(train_loss_arr)+1), train_loss_arr, label = 'Training')
    plt.plot(np.arange(1, len(val_loss_arr)+1), val_loss_arr, label = 'Validation')

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
        if opt.preprocessing_norm is not None:
            preprocessing_norm = opt.preprocessing_norm.capitalize()
        else:
             preprocessing_norm = 'None'
        upper_title = 'Y Preprocessing: {}, Norm: {}'.format(preprocessing, preprocessing_norm)
        train_title = 'Final Training Loss: {}'.format(np.round(train_loss_arr[-1], 3))
        val_title = 'Final Validation Loss: {}'.format(np.round(val_loss_arr[-1], 3))
        plt.title(f'{upper_title}\n{train_title}\n{val_title}', fontsize = 16)


        if opt.milestones is not None:
            lr = float(opt.lr)
            max_y = np.max(np.maximum(train_loss_arr, val_loss_arr))
            min_y = np.min(np.minimum(train_loss_arr, val_loss_arr))
            new_max_y = max_y + (max_y - min_y) * 0.1
            annotate_y = max_y + (max_y - min_y) * 0.05
            x_offset = (opt.milestones[0] - 1) * 0.05
            if not log_y:
                plt.ylim(top = new_max_y)
            plt.axvline(1, linestyle = 'dashed', color = 'green')
            plt.annotate('lr: {}'.format(lr), (1 + x_offset, annotate_y))
            for m in opt.milestones:
                lr = lr * opt.gamma
                plt.axvline(m, linestyle = 'dashed', color = 'green')
                plt.annotate('lr: {:.1e}'.format(lr), (m + x_offset, annotate_y))


    plt.xlabel('Epoch', fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)

    plt.legend()
    plt.tight_layout()
    if log_y:
        plt.yscale('log')
        plt.savefig(osp.join(imagePath, 'train_val_loss_log.png'))
    else:
        plt.savefig(osp.join(imagePath, 'train_val_loss.png'))
    plt.close()

### Functions for plotting sequences ###
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

def plot_seq_exclusive(seq, labels=None, X=None, show = False, save = True,
                        title = None, ofile = 'seq.png'):
    '''Plotting function for mutually exclusive binary particle types'''
    # TODO make figure wider and less tall
    m, k = seq.shape
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))

    for i, c in enumerate(colors):
        x = np.argwhere(seq[:, i] == 1)
        plt.scatter(x, np.ones_like(x), label = i, color = c['color'], s=1)

    if X is not None and labels is not None:
        score = silhouette_score(X, labels)
        lower_title = f'\nsilhouette score: {np.round(score, 3)}'
    else:
        lower_title = ''

    plt.legend()
    ax = plt.gca()
    ax.axes.get_yaxis().set_visible(False)
    if title is not None:
        plt.title(title + lower_title, fontsize=16)
    if save:
        plt.savefig(ofile)
    if show:
        plt.show()
    plt.close()

### Functions for analyzing model performance ###
def analysisIterator(val_dataloader, model, opt, count, mode):
    # format preprocessing title
    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.preprocessing_norm is not None:
        preprocessing_norm = opt.preprocessing_norm.capitalize()
    else:
         preprocessing_norm = 'None'
    upper_title = f'Y Preprocessing: {preprocessing}, Norm: {preprocessing_norm}'

    # format loss title
    assert opt.loss == 'mse'
    loss_title = 'MSE Loss'

    # get samples
    samples = set()
    for i, data in enumerate(val_dataloader):
        if opt.GNN_mode:
            path = data.path[0]
        else:
            path = path[0]
        sample = osp.split(path)[1]
        sample_id = int(sample[6:])
        samples.add(sample_id)

    dataset = get_dataset(opt, True, True, False, samples = samples)

    loss_arr = np.zeros(opt.valN)
    for i, data in enumerate(dataset):
        # get yhat
        assert opt.GNN_mode
        data = data.to(opt.device)
        if opt.output_mode.startswith('energy'):
            y = data.energy
            y = torch.reshape(y, (-1, opt.m, opt.m))
        else:
            y = data.y
            y = torch.reshape(y, (-1, opt.m))
        yhat = model(data)
        path = data.path

        loss = opt.criterion(yhat, y).item()
        y = y.cpu().numpy()
        yhat = yhat.cpu().detach().numpy()
        if opt.output_mode.startswith('energy'):
            y = y.reshape((opt.m, opt.m))
            yhat = yhat.reshape((opt.m, opt.m))
        else:
            y = y.reshape((-1))
            yhat = yhat.reshape((-1))

        loss_arr[i] = loss
        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        if i < count:
            sample = osp.split(path)[1] + '-' + mode
            subpath = osp.join(opt.ofile_folder, sample )
            print(f'{subpath}: {loss}', file = opt.log_file)
            if not osp.exists(subpath):
                os.mkdir(subpath, mode = 0o755)

            yhat_title = '{}\n{} ({}: {})'.format(upper_title, r'$\hat{S}$',
                                                    loss_title, np.round(loss, 3))


            if opt.output_mode.startswith('energy'):
                v_max = np.max(y)
                v_min = np.min(y)

                plot_matrix(yhat, osp.join(subpath, 'energy_hat.png'), vmin = v_min,
                                vmax = v_max, cmap = 'blue-red', title = yhat_title)
                np.savetxt(osp.join(subpath, 'energy_hat.txt'), yhat, fmt = '%.3f')

                plot_matrix(y, osp.join(subpath, 'energy.png'), vmin = v_min,
                                vmax = v_max, cmap = 'blue-red', title = r'$S$')
                np.savetxt(osp.join(subpath, 'energy.txt'), y, fmt = '%.3f')

                # plot dif
                dif = y - yhat
                plot_matrix(dif, osp.join(subpath, 'edif.png'), vmin = -1 * v_max,
                                vmax = v_max, title = r'S - $\hat{S}$', cmap = 'blue-red')
            elif opt.output_mode == 'diag_chi_continuous' or opt.output_mode == 'diag_chi_step':
                plot_diag_chi(None, subpath, y, 'ground_truth', False,
                            'diag_chi_hat.png', yhat, title = f'MSE: {np.round(loss, 3)}',
                            label = 'estimate')
                plot_diag_chi(None, subpath, y, 'ground_truth', True,
                            'diag_chi_hat_log.png', yhat, title = f'MSE: {np.round(loss, 3)}',
                            label = 'estimate')

                np.savetxt(osp.join(subpath, 'diag_chi.txt'), y, fmt = '%.3f')
                np.savetxt(osp.join(subpath, 'diag_chi_hat.txt'), yhat, fmt = '%.3f')
            else:
                return

            # tar subpath
            os.chdir(opt.ofile_folder)
            with tarfile.open(f'{sample}.tar.gz', 'w:gz') as f:
                f.add(sample)
            rmtree(sample)

    print(f'Loss: {np.mean(loss_arr)} +- {np.std(loss_arr)}\n',
        file = opt.log_file)

def plotEnergyPredictions(val_dataloader, model, opt, count = 5):
    print('Prediction Results:', file = opt.log_file)
    assert opt.output_mode.startswith('energy')
    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.preprocessing_norm is not None:
        preprocessing_norm = opt.preprocessing_norm.capitalize()
    else:
         preprocessing_norm = 'None'
    upper_title = f'Y Preprocessing: {preprocessing}, Norm: {preprocessing_norm}'

    assert opt.loss == 'mse'
    loss_title = 'MSE Loss'

    loss_arr = np.zeros(min(count, opt.valN))
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        assert opt.GNN_mode
        data = data.to(opt.device)
        y = data.energy
        y = torch.reshape(y, (-1, opt.m, opt.m))
        yhat = model(data)
        path = data.path[0]

        loss = opt.criterion(yhat, y).item()
        y = y.cpu().numpy().reshape((opt.m, opt.m))
        yhat = yhat.cpu().detach().numpy().reshape((opt.m,opt.m))

        sample = osp.split(path)[-1]
        subpath = osp.join(opt.ofile_folder, sample)
        print('{}: {}'.format(subpath, loss), file = opt.log_file)
        if not osp.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        yhat_title = '{}\n{} ({}: {})'.format(upper_title, r'$\hat{S}$',
                                                loss_title, np.round(loss, 3))
        loss_arr[i] = loss
        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        v_max = np.max(y)
        v_min = np.min(y)

        plot_matrix(yhat, osp.join(subpath, 'energy_hat.png'), vmin = v_min,
                        vmax = v_max, cmap = 'blue-red', title = yhat_title)
        np.savetxt(osp.join(subpath, 'energy_hat.txt'), yhat, fmt = '%.3f')

        plot_matrix(y, osp.join(subpath, 'energy.png'), vmin = v_min,
                        vmax = v_max, cmap = 'blue-red', title = r'$S$')
        np.savetxt(osp.join(subpath, 'energy.txt'), y, fmt = '%.3f')

        # plot dif
        dif = y - yhat
        plot_matrix(dif, osp.join(subpath, 'edif.png'), vmin = -1 * v_max,
                        vmax = v_max, title = r'S - $\hat{S}$', cmap = 'blue-red')

        # plot plaid contribution
        plaid_hat = model.plaid_component(data)
        if plaid_hat is not None:
            plaid_hat = plaid_hat.cpu().detach().numpy().reshape((opt.m,opt.m))
            plot_matrix(plaid_hat, osp.join(subpath, 'plaid_hat.png'), vmin = -1 * v_max,
                            vmax = v_max, title = 'plaid portion', cmap = 'blue-red')

        # plot diag contribution
        diagonal_hat = model.diagonal_component(data)
        if diagonal_hat is not None:
            diagonal_hat = diagonal_hat.cpu().detach().numpy().reshape((opt.m,opt.m))
            plot_matrix(diagonal_hat, osp.join(subpath, 'diagonal_hat.png'), vmin = -1 * v_max,
                            vmax = v_max, title = 'diagonal portion', cmap = 'blue-red')


        # tar subpath
        os.chdir(opt.ofile_folder)
        print(os.getcwd())
        with tarfile.open(f'{sample}.tar.gz', 'w:gz') as f:
            f.add(sample)
        rmtree(sample)

    print(f'Loss: {np.mean(loss_arr)} +- {np.std(loss_arr)}\n',
        file = opt.log_file)

def plotDiagChiPredictions(val_dataloader, model, opt, count = 5):
    print('Prediction Results:', file = opt.log_file)

    loss_arr = np.zeros(min(count, opt.valN))
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        if opt.GNN_mode:
            data = data.to(opt.device)
            yhat = model(data)
            path = data.path[0]
        else:
            x, _, path = data
            x = x.to(opt.device)
            path = path[0]
            yhat = model(x)
        yhat = yhat.cpu().detach().numpy().reshape((-1))

        y = np.load(osp.join(path, 'diag_chis_continuous.npy'))
        if opt.crop is not None:
            y = y[opt.crop[0]:opt.crop[1]]

        # format yhat
        if 'bond_length' in opt.output_mode:
            bond_length_hat = yhat[-1]
            yhat = yhat[:-1]
        if opt.output_mode == 'diag_chi_continuous' or opt.output_mode == 'diag_chi_step':
            # no need to do anything
            pass
        elif opt.output_mode.startswith('diag_chi'):
            with open(osp.join(path, 'config.json'), 'r') as f:
                config = json.load(f)
            yhat = calculate_diag_chi_step(config, yhat)
        elif opt.output_mode.startswith('diag_param'):
            d_arr = np.arange(len(y))
            with open(osp.join(path, 'params.log'), 'r') as f:
                line = f.readline()
                while line != '':
                    line = f.readline()
                    if line.startswith('Diag chi args:'):
                        line = f.readline().split(', ')
                        for arg in line:
                            if arg.startswith('diag_chi_method'):
                                method = arg.split('=')[1].strip("'")
            if method == 'logistic':
                print('predicted params:', yhat)
                min_val, max_val, slope, midpoint = yhat
                yhat = (max_val - min_val)/(1 + np.exp(-1*slope * (d_arr - midpoint))) + min_val
            elif method == 'log':
                scale, slope, constant = yhat
                yhat = scale * np.log(slope * d_arr + 1) + args.constant

        loss = mean_squared_error(y, yhat)
        loss_arr[i] = loss
        if opt.verbose:
            print('y', y)
            print('yhat', yhat)

        sample = osp.split(path)[-1]
        subpath = osp.join(opt.ofile_folder, sample)
        print(f'{subpath}: {loss}', file = opt.log_file)
        if not osp.exists(subpath):
            os.mkdir(subpath, mode = 0o755)

        plot_diag_chi(None, subpath, y, 'ground_truth', False,
                    'diag_chi_hat.png', yhat, title = f'MSE: {np.round(loss, 3)}',
                    label = 'estimate')
        plot_diag_chi(None, subpath, y, 'ground_truth', True,
                    'diag_chi_hat_log.png', yhat, title = f'MSE: {np.round(loss, 3)}',
                    label = 'estimate')

        np.savetxt(osp.join(subpath, 'diag_chi.txt'), y, fmt = '%.3f')
        np.savetxt(osp.join(subpath, 'diag_chi_hat.txt'), yhat, fmt = '%.3f')

    print(f'Loss: {np.mean(loss_arr)} +- {np.std(loss_arr)}\n',
        file = opt.log_file)

def downsamplingAnalysis(val_dataloader, model, opt, count = 5):
    print('Downsampling Results:', file = opt.log_file)
    opt_copy = copy.copy(opt) # shallow copy only
    if opt_copy.root_name is not None:
        opt_copy.root_name += 'downsample'
    opt_copy.y_preprocessing = 'sweep200000_' + opt_copy.y_preprocessing

    analysisIterator(val_dataloader, model, opt_copy, count, 'downsampling')

def rescalingAnalysis(val_dataloader, model, opt, count = 5):
    print('Rescaling Results:', file = opt.log_file)
    opt_copy = copy.copy(opt) # shallow copy only
    if opt_copy.root_name is not None:
        opt_copy.root_name += 'rescale'
    opt_copy.y_preprocessing = 'rescale2_' + opt_copy.y_preprocessing

    analysisIterator(val_dataloader, model, opt_copy, count, 'rescaling')

#### Functions for plotting xyz files ####
def plot_xyz(xyz, L, x = None, ofile = None, show = True, title = None, legend = True):
    '''
    Plots particles in xyz as 3D scatter plot.
    Only supports mutually exclusive bead types for coloring. # TODO
    Inputs:
        xyz: shape (N,3) array of all particle positions
        L: side of LxLxL box (nm), if None the plot will be fit to the input
        x: bead types to color
        LJ: True if Lennard Jones particles
        ofile: location to save image
        show: True to show
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

def plot_xyz_gif(xyz, x, dir, ofile = 'xyz.gif', order = None):
    filenames = []
    if order is None:
        order = range(len(xyz))
    for i in order:
        fname = osp.join(dir, f'{i}.png')
        filenames.append(fname)
        plot_xyz(xyz[i, :, :], None, x = x, ofile = fname, show = False,
                    title = None, legend = False)

    # build gif
    # filenames = [osp.join(dir, f'ovito0{i}.png') for i in range(100, 900)]
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(osp.join(dir, ofile), frames, format='GIF', fps=2)

    # remove files
    for filename in set(filenames):
        os.remove(filename)

def plot_sc_contact_maps(dataset, samples, ofolder = 'sc_contact', count = 20,
                        jobs = 1, overall = True, N_max = None, crop_size = None,
                        correct_diag = False, sparsify = False):
    if isinstance(samples, int):
        samples = [samples]

    for sample in samples:
        print(f'sample{sample}')
        dir = osp.join(dataset, 'samples', f'sample{sample}')
        odir = osp.join(dir, ofolder)

        sc_contacts = load_sc_contacts(dir, zero_diag = True, jobs = jobs, triu = True,
                                        N_max = N_max, correct_diag = correct_diag,
                                        sparsify = sparsify)
        plot_sc_contact_maps_inner(sc_contacts, odir, count, jobs, overall, crop_size)

def plot_sc_contact_maps_inner(sc_contacts, odir, count, jobs, overall = False,
                                crop_size = None, vmax = 'mean', save_txt = False,
                                title_index = False):
    '''
    Plot sc contact maps.

    Inputs:
        sc_contacts: np array of sc contact maps (in full or flattened upper traingle)
        odir: directory to write to
        count: number of plots to make
        jobs: number of jobs for plotting in parallel
        overall: True to plot overall contact map
        crop_size: size to crop contact map
        vmax: vmax from plot_matrix
        save_txt: True to save txt file
        title_index: True to title with index
    '''
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o775)

    if sp.issparse(sc_contacts):
        sc_contacts = sc_contacts.toarray()

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

    if crop_size is not None:
        overall_m = min([crop_size, m])
    else:
        overall_m = m
    overall_map = np.zeros((overall_m,overall_m))
    mapping = []
    filenames = []
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
            if save_txt:
                np.savetxt(osp.join(odir, f'{i}.txt'), contact_map, fmt='%.2f')
            filenames.append(f'{i}.png')
            title = i if title_index else None
            if jobs > 1:
                mapping.append((contact_map, osp.join(odir, f'{i}.png'), title, 0, vmax))
            else:
                plot_matrix(contact_map, osp.join(odir, f'{i}.png'), title = title, vmax = vmax)

        if overall:
            overall_map += contact_map

    if jobs > 1:
        with multiprocessing.Pool(jobs) as p:
            p.starmap(plot_matrix, mapping)

    if overall:
        plot_matrix(overall_map, osp.join(odir, 'overall.png'), vmax = 'abs_max')

    return filenames

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

#### Functions for plotting diag chis and contact probability curves ####
def plot_diag_chi(config, path, ref = None, ref_label = '', logx = False,
                ofile = None, diag_chis_step = None, ylim = (None, None),
                title = None, label = ''):
    '''
    config: config file
    path: save file path
    ref: reference parameters
    ref_label: label for reference parameters
    '''
    if config is None:
        assert diag_chis_step is not None
    else:
        diag_chis_step = calculate_diag_chi_step(config)

    fig, ax = plt.subplots()
    ax.plot(diag_chis_step, color = 'k', label = label)
    ax.set_xlabel('Polymer Distance', fontsize = 16)
    ax.set_ylabel('Diagonal Parameter', fontsize = 16)
    if ref is not None:
        if isinstance(ref, np.ndarray):
            pass
        elif osp.exists(ref):
            ref = np.load(ref)
        ax.plot(ref, color = 'k', ls = '--', label = ref_label)
        if ref_label != '':
            plt.legend()

    ax.set_ylim(ylim[0], ylim[1])
    if logx:
        ax.set_xscale('log')
    if title is not None:
        plt.title(title)

    if ofile is None:
        if logx:
            ofile = osp.join(path, 'chi_diag_step_log.png')
        else:
            ofile = osp.join(path, 'chi_diag_step.png')
    else:
        ofile = osp.join(path, ofile)

    plt.savefig(ofile)
    plt.close()

    return diag_chis_step

def plot_mean_vs_genomic_distance(y, path, ofile, diag_chis_step = None,
                                config = None, logx = False, ref = None,
                                ref_label = 'reference'):
    '''
    Wrapper for plot_mean_dist that takes contact map as input.

    Inputs:
        y: contact map
        path: save path
        ofile: save file name
        diag_chis_step: diagonal chi parameter as function of d (None to skip)
        config: config file (None to skip)
        logx: True to log-scale x axis
    '''
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                            zero_diag = False, zero_offset = 0)
    if config is not None:
        diag_chis_step = calculate_diag_chi_step(config)

    plot_mean_dist(meanDist, path, ofile, diag_chis_step, logx, ref, ref_label)

    return meanDist

def plot_mean_dist(meanDist, path, ofile, diag_chis_step, logx, ref,
                    ref_label = 'reference', label = '', color = 'blue',
                    title = None):
    '''
    Inputs:
        meanDist: contact map
        path: save path
        ofile: save file name
        diag_chis_step: diagonal chi parameter as function of d (None to skip)
        logx: True to log-scale x axis
    '''
    meanDist = meanDist.copy()
    if ref is not None:
        ref = ref.copy()

    fig, ax = plt.subplots()
    if ref is not None:
        ax.plot(ref, label = ref_label, color = 'k')
    ax.plot(meanDist, label = label, color = color)
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if diag_chis_step is not None:
        ax2 = ax.twinx()
        ax2.plot(diag_chis_step, ls = '--', label = 'Parameters', color = color)
        ax2.set_ylabel('Diagonal Parameter', fontsize = 16)
        if logx:
            ax2.set_xscale('log')
        ax2.legend(loc='upper right')
    # else:
    #     x = np.arange(1, 10).astype(np.float64)
    #     ax.plot(x, np.power(x, -1)/4, color = 'grey', ls = '--', label='-1')
    #     x = np.arange(10, 100).astype(np.float64)
    #     ax.plot(x, np.power(x, -1.5), color = 'grey', ls = ':', label='-3/2')
    #     ax.legend(loc='upper right')

    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(osp.join(path, ofile))
    plt.close()

### Primary scripts ###
def plot_matrix(arr, ofile = None, title = None, vmin = 0, vmax = 1,
                    size_in = 6, minVal = None, maxVal = None, prcnt = False,
                    cmap = None, x_ticks = None, y_ticks = None):
    """
    Plotting function for 2D arrays.

    Inputs:
        arr: numpy array
        ofile: save location (None to show instead)
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
    elif cmap.replace('-', '').lower() == 'bluered':
        if vmin == 'min' and vmax == 'max':
            vmin = vmax = 'center'
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0, 'blue'),
                                                 (0.5, 'white'),
                                                  (1, 'red')], N=126)
    else:
        raise Exception('Invalid cmap: {}'.format(cmap))

    if len(arr.shape) == 4:
        N, C, H, W = arr.shape
        assert N == 1 and C == 1
        arr = arr.reshape(H,W)
    elif len(arr.shape) == 3:
        N, H, W = arr.shape
        assert N == 1
        y = arr.reshape(H,W)
    else:
        H, W = arr.shape

    if minVal is not None or maxVal is not None:
        arr = arr.copy() # prevent issues from reference type
    if minVal is not None:
        ind = arr < minVal
        arr[ind] = 0
    if maxVal is not None:
        ind = arr > maxVal
        arr[ind] = 0
    plt.figure(figsize = (size_in, size_in))

    # set min and max
    if vmin == 'center':
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
    elif vmin == 'center1':
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        d_vmin = 1-vmin
        d_vmax = vmax-1
        d = max(d_vmax, d_vmin)
        vmin = 1 - d
        vmax = 1 + d
    else:
        if vmin == 'min':
            vmin = np.nanpercentile(arr, 1)
            print(vmin)
            # uses 1st percentile instead of absolute min
        elif vmin == 'abs_min':
            vmin = np.min(arr)

        if vmax == 'mean':
            vmax = np.mean(arr)
        elif vmax == 'max':
            vmax = np.nanpercentile(arr, 99)
            # uses 99th percentile instead of absolute max
        elif vmax == 'abs_max':
            vmax = np.max(arr)

    ax = sns.heatmap(arr, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap)
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

def plot_matrix_gif(arr, dir, ofile = None, title = None, vmin = 0, vmax = 1,
                    size_in = 6, minVal = None, maxVal = None, prcnt = False,
                    cmap = None, x_ticks = None, y_ticks = None):
    filenames = []
    for i in range(len(arr)):
        fname=osp.join(dir, f'{i}.png')
        filenames.append(fname)
        plot_matrix(arr[i,], fname, ofile, title, vmin, vmax, size_in, minVal,
                    maxVal, prcnt, cmap, x_ticks, y_ticks)

    # build gif
    # filenames = [osp.join(dir, f'ovito0{i}.png') for i in range(100, 900)]
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(ofile, frames, format='GIF', fps=1)

    # remove files
    for filename in set(filenames):
        os.remove(filename)

def plotting_script(model, opt, train_loss_arr = None, val_loss_arr = None,
                    dataset = None):
    if model is None:
        model, train_loss_arr, val_loss_arr = load_saved_model(opt, verbose = True, throw = False)
    if model is not None and dataset is None:
        dataset = get_dataset(opt, True, True)

    opt.batch_size = 1 # batch size must be 1
    opt.shuffle = False # for reproducibility
    _, val_dataloader, _ = get_data_loaders(dataset, opt)



    imagePath = opt.ofile_folder
    print('#### Plotting Script ####', file = opt.log_file)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt)
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt, True)

    if opt.plot_predictions:
        if opt.output_mode.startswith('energy'):
            plotEnergyPredictions(val_dataloader, model, opt)
        elif opt.output_mode.startswith('diag'):
            plotDiagChiPredictions(val_dataloader, model, opt)

    if opt.plot:
        if not opt.y_preprocessing.startswith('sweep'):
            downsamplingAnalysis(val_dataloader, model, opt)
        rescalingAnalysis(val_dataloader, model, opt)
