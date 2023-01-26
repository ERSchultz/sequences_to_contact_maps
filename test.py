import json
import os
import os.path as osp
import time
from shutil import copyfile, rmtree

import cooler
import hicrep.utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core_test_train import core_test_train
from result_summary_plots import plot_top_PCs
from scipy import linalg
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scripts.argparse_utils import (ArgparserConverter, finalize_opt,
                                    get_base_parser)
from scripts.energy_utils import *
from scripts.load_utils import load_sc_contacts, save_sc_contacts
from scripts.neural_nets.dataset_classes import make_dataset
from scripts.neural_nets.networks import get_model
from scripts.neural_nets.utils import get_dataset
from scripts.plotting_utils import plot_matrix
from scripts.similarity_measures import SCC
from scripts.utils import (DiagonalPreprocessing, calc_dist_strat_corr, crop,
                           print_time, triu_to_full)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def test_num_workers():
    opt = argparseSetup() # get default args
    opt.data_folder = "/project2/depablo/erschultz/dataset_04_18_21"
    opt.cuda = True
    opt.device = torch.device('cuda')
    opt.y_preprocessing = 'diag'
    opt.preprocessing_norm = 'batch'
    dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.y_preprocessing,
                                        opt.preprocessing_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop)

    b_arr = np.array([1, 2, 4, 8, 16, 32])
    w_arr = np.array([0,1,2,3,4,5,6,7,8])
    results = np.zeros((len(b_arr), len(w_arr)))
    for i, b in enumerate(b_arr):
        print(b)
        for j, w in enumerate(w_arr):
            t0 = time.time()
            opt.batch_size = int(b)
            opt.num_workers = w
            _, val_dataloader, _ = get_dataLoaders(dataset, opt)
            for x, y in val_dataloader:
                x = x.to(opt.device)
                y = y.to(opt.device)
            results[i, j] = time.time() - t0

    print(np.round(results, 1))

def edit_argparse():
    dir = "results"
    for type in os.listdir(dir):
        type_path = osp.join(dir, type)
        if not 'Energy' in type_path:
            continue
        print(type_path)
        if osp.isdir(type_path):
            for id in os.listdir(type_path):
                id_path = osp.join(type_path, id)
                if osp.isdir(id_path):
                    print(id_path)
                    arg_file = osp.join(id_path, 'argparse.txt')
                    if osp.exists(arg_file):
                        with open(arg_file, 'r') as f:
                            lines = f.readlines()
                        new_lines = []
                        i = 0
                        split = False
                        while i < len(lines):
                            line = lines[i]
                            if line.startswith('--relabel'):
                                assert lines.pop(i+1).strip() == 'false'
                                lines.pop(i)
                            if line.startswith('--split_edges_for_feature_augmentation'):
                                if lines.pop(i+1).strip() == 'true':
                                    split = True
                                else:
                                    split = False
                                lines.pop(i)

                                i -= 2
                                assert lines[i].startswith('--pre_transforms')
                                i += 1
                                line = lines[i]

                                line_split = line.strip().split('-')
                                for j, val in enumerate(line_split):
                                    if val.lower().startswith('degree'):
                                        if split and 'split' not in val:
                                            line_split[j] = val + '_split'
                                line = '-'.join(line_split)+'\n'
                                lines[i] = line
                            if line.startswith('--pre_transforms'):
                                i += 1
                                line = lines[i]
                                line_split = line.strip().split('-')
                                degree_count = 0
                                split = False
                                for j, val in enumerate(line_split):
                                    if val.lower().startswith('degree'):
                                        print(val)
                                        degree_count += 1
                                        if 'split' in val:
                                            print('h', val)
                                            split = True
                                if degree_count == 1 and split:
                                    print('j')
                                    line_split.insert(0, 'Degree')
                                line = '-'.join(line_split)+'\n'

                                lines[i] = line

                            i += 1
                        with open(arg_file, 'w') as f:
                            f.write("".join(lines))

def debugModel(model_type):
    parser = get_base_parser()
    AC = ArgparserConverter()
    opt = parser.parse_args()

    # dataset
    dir = "/home/erschultz"
    datasets = ['dataset_11_18_22']
    opt.data_folder = [osp.join(dir, d) for d in datasets]
    opt.scratch = '/home/erschultz/scratch'

    # architecture
    opt.m = 1024
    opt.split_percents=[0.8,0.2,0.0]
    opt.split_sizes=None
    # opt.split_sizes=[1, 0, 0]
    # opt.split_percents = None
    opt.random_split=True

    if model_type == 'Akita':
        opt.kernel_w_list=AC.str2list('5-5-5')
        opt.hidden_sizes_list=AC.str2list('4-6-8')
        opt.dilation_list_trunk=AC.str2list('2-4-8-16')
        opt.bottleneck=4
        opt.dilation_list_head=AC.str2list('2-4-8-16')
        opt.out_act=nn.ReLU(True)
        opt.training_norm='batch'
        opt.down_sampling='conv'
    elif model_type == 'UNet':
        opt.toxx = True
        opt.toxx_mode = 'mean'
        opt.x_reshape = False
        opt.nf = 8
        opt.out_act = 'sigmoid'
        opt.training_norm = 'batch'
    elif model_type == 'DeepC':
        opt.k=10
        opt.preprocessing_norm=None
        opt.y_preprocessing='diag'
        opt.kernel_w_list=AC.str2list('5-5-5')
        opt.hidden_sizes_list=AC.str2list('32-64-128')
        opt.dilation_list=AC.str2list('32-64-256')
    elif model_type == 'ContactGNN':
        opt.GNN_mode = True
        opt.output_mode = 'sequence'
        opt.loss = 'BCE'
        opt.preprocessing_norm = None
        opt.message_passing='SignedConv'
        opt.hidden_sizes_list=AC.str2list('16-2')
        opt.out_act = None
        opt.use_node_features = False
        opt.use_edge_weights = False
        opt.transforms=AC.str2list('none')
        opt.pre_transforms=AC.str2list('degree')
        opt.split_neg_pos_edges_for_feature_augmentation = True
        opt.top_k = None
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.relabel_11_to_00 = False
        opt.log_preprocessing = 'True'
        opt.head_architecture = 'fc'
        opt.head_hidden_sizes_list = [2]
        # opt.crop=[50,100]
        # opt.m = 50
        # opt.use_bias = False
    elif model_type == 'ContactGNNEnergy':
        opt.y_preprocessing = 'sweeprand_log_inf'
        opt.rescale = 4
        opt.mean_filt = None
        opt.kr = False
        opt.keep_zero_edges = False
        opt.loss = 'mse'
        opt.preprocessing_norm = 'mean'
        opt.message_passing = 'weighted_GAT'
        opt.GNN_mode = True
        opt.output_mode = 'energy_sym_diag'
        opt.output_preprocesing = 'log'
        opt.encoder_hidden_sizes_list=[32]
        opt.edge_encoder_hidden_sizes_list=[16]
        opt.update_hidden_sizes_list=[100,64]
        opt.hidden_sizes_list=[8]
        opt.gated = False
        opt.dropout = 0.0
        opt.act = 'prelu'
        opt.inner_act = 'relu'
        opt.out_act = 'relu'
        opt.head_act = 'relu'
        opt.training_norm = None
        opt.use_node_features = False
        opt.k = 8
        opt.use_edge_weights = False
        opt.use_edge_attr = True
        # opt.transforms=AC.str2list('constant')
        opt.pre_transforms=AC.str2list('constant-ContactDistance-GeneticDistance_norm')
        opt.mlp_model_id=None
        opt.sparsify_threshold = None
        opt.sparsify_threshold_upper = None
        opt.log_preprocessing = None
        opt.head_architecture = 'dconv-bilinear-triu'
        opt.head_architecture_2 = 'dconv-fc-fill_1024'
        opt.head_hidden_sizes_list = [1000, 1000]
        opt.crop = None

        opt.use_bias = True
        opt.num_heads = 8
        opt.concat_heads = True
        # opt.max_diagonal=500
    elif model_type.lower() == 'signnet':
        model_type = 'ContactGNNEnergy'
        opt.use_sign_plus = True
        opt.y_preprocessing = 'sweeprand_log_inf'
        opt.rescale = None
        opt.loss = 'mse'
        opt.preprocessing_norm = 'mean'
        opt.message_passing = 'weighted_GAT'
        opt.GNN_mode = True
        opt.output_mode = 'energy_sym_diag'
        opt.output_preprocesing = 'log'
        opt.encoder_hidden_sizes_list=AC.str2list('none')
        # opt.edge_encoder_hidden_sizes_list=[100,100,3]
        opt.update_hidden_sizes_list=[100,100,64]
        opt.hidden_sizes_list=[8]
        opt.act = 'prelu'
        opt.inner_act = 'relu'
        opt.out_act = 'relu'
        opt.head_act = 'relu'
        opt.use_edge_attr = True
        # opt.transforms=AC.str2list('constant')
        opt.pre_transforms=AC.str2list('constant-ContactDistance-GeneticDistance_norm-AdjPCs_8-AdjPCA_3')
        opt.k=8
        opt.head_architecture = 'bilinear_triu'
        opt.head_architecture_2 = 'fc-fill_1024'
        opt.head_hidden_sizes_list = [1000, 1000, 1000]
        # opt.crop = [0,21]

        opt.use_bias = True
        opt.num_heads = 8
        opt.concat_heads = True
    elif model_type == 'ContactGNNDiag':
        opt.loss = 'mse'
        opt.preprocessing_norm = 'instance'
        opt.message_passing='gat'
        opt.GNN_mode = True
        opt.output_mode = 'diag_chi'
        opt.encoder_hidden_sizes_list=None
        opt.update_hidden_sizes_list=None
        opt.hidden_sizes_list=[3,3]
        opt.training_norm = 'instance'
        opt.use_edge_weights = False
        opt.use_edge_attr = True
        opt.transforms=AC.str2list('empty')
        opt.pre_transforms=AC.str2list('degree-contactdistance')
        opt.split_edges_for_feature_augmentation = False
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.log_preprocessing = None
        opt.head_architecture = 'concat'
        opt.head_architecture_2 = 'bilinear'
        opt.head_hidden_sizes_list = [1000, 1000, 1000, 1]
        opt.use_bias = True
        opt.use_bias = True
        opt.num_heads = 2
    elif model_type == 'MLP':
        opt.scratch = '/home/erschultz/scratch'
        opt.preprocessing_norm=None
        opt.y_preprocessing='log'
        opt.hidden_sizes_list=AC.str2list('100-'*6 + '1024')
        opt.act='prelu'
        opt.out_act='prelu'
        opt.output_mode='diag_chi_continuous'
        # opt.scheduler='multisteplr'
        opt.scheduler='reducelronplateau'
        opt.min_lr=1e-4
        # opt.milestones=[2,4]
        opt.log_preprocessing=None
        opt.y_zero_diag_count=0
        # opt.training_norm='batch'
        opt.dropout=False
        opt.dropout_p=0.1
        # opt.crop=[20,512]
        # opt.m = 980

    # hyperparameters
    opt.n_epochs = 2
    opt.lr = 1e-3
    opt.weight_decay = 1e-5
    opt.w_reg = None
    opt.reg_lambda = 10
    opt.batch_size = 1
    opt.gamma = 0.1

    # other
    opt.plot = False
    opt.plot_predictions = False
    opt.verbose = False
    opt.print_params = True
    opt.gpus = 1
    # opt.delete_root = True
    opt.num_workers = 2
    opt.use_scratch_parallel = False

    opt = finalize_opt(opt, parser, False, debug = True)
    opt.model_type = model_type
    model = get_model(opt)
    core_test_train(model, opt)


    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    # for val, label in zip([False, True], ['False', 'True']):
    #     opt.kr = val
    #
    #     print(opt, end = '\n\n', file = opt.log_file)
    #     dataset = get_dataset(opt)
    #     for i, data in enumerate(dataset):
    #         print(data.path)
    #         # print(data.contact_map_diag, torch.min(data.contact_map_diag))
    #         # plot_matrix(data.contact_map_diag, osp.join(data.path, 'diag.png'), title = None, cmap='bluered', vmin = 'center1')
    #         print(f'x={data.x}, shape={data.x.shape}, '
    #                 f'min={torch.min(data.x).item()}, '
    #                 f'max={torch.max(data.x).item()}')
    #         print(f'edge_attr={data.edge_attr}, '
    #                 f'shape={data.edge_attr.shape}, '
    #                 f'min={torch.min(data.edge_attr).item()}, '
    #                 f'max={torch.max(data.edge_attr).item()}')
    #         ax3.hist(data.edge_attr.reshape(-1), alpha = 0.5, label = f'{label}',
    #                 bins=50)
    #         ax0.hist(data.x[:,0], alpha = 0.5, label = f'{label}',
    #                 bins=50)
    #         ax1.hist(data.x[:,1], alpha = 0.5, label = f'{label}',
    #                 bins=50)
    #         ax2.hist(data.x[:,2], alpha = 0.5, label = f'{label}',
    #                 bins=50)
    # # plt.yscale('log')
    # for ax in [ax0, ax1, ax2, ax3]:
    #     ax.set_yscale('log')
    #
    # ax0.set_ylabel('Count')
    # ax0.set_title('deg')
    # ax1.set_title('pos')
    # ax2.set_title('neg')
    # ax3.set_title('attr')
    # plt.legend()
    # plt.show()

    if opt.move_data_to_scratch:
        rmtree(opt.data_folder)

def binom():
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_04_27_22/samples/sample1'
    # dir = '/home/erschultz/dataset_test/samples/sample9'
    y = np.load(osp.join(dir, 'y_diag.npy'))
    m = len(y)
    s_sym = np.load(osp.join(dir, 's_sym.npy'))
    y = y[np.triu_indices(m)]
    n = np.max(y)

    y_full = triu_to_full(y)
    pca = PCA()
    pca.fit(y_full)
    Vt = pca.components_
    for j in range(10):
        min = np.min(Vt[j])
        max = np.max(Vt[j])
        if max > abs(min):
            val = max
        else:
            val = abs(min)
        # multiply by scale such that val x scale = 1
        scale = 1/val
        Vt[j] *= scale
    optimal_pca_dict = {}

    p = y/n
    loss = binom_nll(p, y, n, 100)
    print('p loss', loss)
    p = triu_to_full(p)
    plot_matrix(p, osp.join(dir, 'p.png'), title = None, vmin = 'min', vmax = 'max')

    for i in [2,3,4,5,6,8,10]:
        print(f'\nk={i}')
        pca = PCA(n_components = i)
        s_transform = pca.fit_transform(s_sym)
        s_sym_i = pca.inverse_transform(s_transform)

        # is pca the same as s?
        maxent_dir = osp.join(dir, f'PCA-normalize/k{i}/replicate1')
        s_file = osp.join(maxent_dir, 's_sym.npy')
        if osp.exists(s_file):
            s_sym_pca = np.load(s_file)
            print('min', np.min(s_sym_pca), 'max', np.max(s_sym_pca))

            plot_binom_helper(s_sym_pca[np.triu_indices(m)],
                                s_sym_i[np.triu_indices(m)],
                                r'$sPCA_{ij}$',
                                r'$s_{ij}$',
                                osp.join(dir, f's_sym_pca_{i} vs s_sym_{i}.png'),
                                title = f'\nMSE: {np.round(mean_squared_error(s_sym_pca, s_sym_i), 3)}')
            plot_binom_helper2(s_sym_pca,
                                s_sym_i,
                                r'$\hat{s}_{ij}$',
                                r'$s_{ij}$',
                                osp.join(dir, f's_sym_pca_{i}_s vs s_sym_{i}.png'))

            plot_binom_helper(s_sym_pca[np.triu_indices(m)],
                                s_sym[np.triu_indices(m)],
                                r'$sPCA_{ij}$',
                                r'$s_{ij}$',
                                osp.join(dir, f's_sym_pca_{i} vs s_sym.png'),
                                title = f'\nMSE: {np.round(mean_squared_error(s_sym_pca, s_sym), 3)}')
            plot_binom_helper2(s_sym_pca,
                                s_sym,
                                r'$\hat{s}_{ij}$',
                                r'$s_{ij}$',
                                osp.join(dir, f's_sym_pca_{i}_s vs s_sym.png'))
        else:
            s_sym_pca = None

        # optimal s_pca
        if i in optimal_pca_dict:
            s_sym_pca_min = optimal_pca_dict[i]
        else:
            v = Vt[:i].T
            model = minimize(s_sym_pca_mse, np.zeros((i,i))[np.triu_indices(i)], args = (v, s_sym))
            print(model)
            s = v @ triu_to_full(model.x) @ v.T
            s_sym_pca_min = (s + s.T)/2
        plot_matrix(s_sym_pca_min, osp.join(dir, f's_pca_{i}_min_MSE.png'),
                        vmin = 'min', vmax = 'max', cmap = 'bluered')
        np.save(osp.join(dir, f's_pca_{i}_min_MSE.npy'), s_sym_pca_min)

        # is optimal pca the same as s?
        plot_binom_helper(s_sym_pca_min[np.triu_indices(m)],
                            s_sym[np.triu_indices(m)],
                            r'$sPCA_{ij}-min$',
                            r'$s_{ij}$',
                            osp.join(dir, f's_sym_pca_{i}_min vs s_sym.png'),
                            title = f'\nMSE: {np.round(mean_squared_error(s_sym_pca_min, s_sym), 3)}')
        plot_binom_helper2(s_sym_pca_min,
                            s_sym,
                            r'$\hat{s}_{ij}$',
                            r'$s_{ij}$',
                            osp.join(dir, f's_sym_pca_{i}_min_s vs s_sym.png'))

        plot_binom_helper(s_sym_pca_min[np.triu_indices(m)],
                            s_sym_i[np.triu_indices(m)],
                            r'$sPCA_{ij}-min$',
                            r'$s_{ij}$',
                            osp.join(dir, f's_sym_pca_{i}_min vs s_sym_{i}.png'),
                            title = f'\nMSE: {np.round(mean_squared_error(s_sym_pca_min, s_sym_i), 3)}')
        plot_binom_helper2(s_sym_pca_min,
                            s_sym_i,
                            r'$\hat{s}_{ij}$',
                            r'$s_{ij}$',
                            osp.join(dir, f's_sym_pca_{i}_min_s vs s_sym_{i}.png'))

        # pca vs optimal pca
        plot_binom_helper(s_sym_pca[np.triu_indices(m)],
                            s_sym_pca_min[np.triu_indices(m)],
                            r'$sPCA_{ij}$',
                            r'$sPCA_{ij}$-min',
                            osp.join(dir, f's_sym_pca_{i} vs s_sym_pca_{i}_min.png'),
                            title = f'\nMSE: {np.round(mean_squared_error(s_sym_pca, s_sym_pca_min), 5)}')
        plot_binom_helper2(s_sym_pca,
                            s_sym_pca_min,
                            r'$\hat{s}_{ij}$',
                            r'$sPCA_{ij}$-min',
                            osp.join(dir, f's_sym_pca_{i}_s vs s_sym_pca_{i}_min.png'))


    # print('-'*30)
    # lmbda = 100
    # model = minimize(binom_nll, np.ones_like(y) / 2, args = (y, n, lmbda),
    #                 bounds = [(0,1)]*len(y))
    # print(model)

def binom_nll(p, y, n, lmbda = 100):
    ll = 0
    l = 0
    for i in range(len(p)):
        # print(p[i], y[i])
        left = p[i]**y[i]
        right = (1 - p[i])**(n - y[i])
        likelihood = left*right
        l += likelihood
        # ll += np.log(likelihood)
    nll = -1 * np.log(l/(1+l))

    p_full = triu_to_full(p)
    reg = lmbda * np.linalg.norm(p_full, 'nuc')

    return likelihood + reg

def s_sym_pca_mse(chi, v, s_sym):
    s = v @ triu_to_full(chi) @ v.T

    s_sym_pca = (s+s.T)/2
    mse = mean_squared_error(s_sym_pca, s_sym)
    return mse

def plot_binom_helper(x, y, xlabel, ylabel, ofile, density = False, title = ''):
    reg = LinearRegression()
    x_new = x.reshape(-1, 1)
    y_new = y.reshape(-1, 1)
    reg.fit(x_new, y_new)
    score = np.round(reg.score(x_new, y_new), 3)
    yhat = triu_to_full(reg.predict(x_new).reshape(-1))

    if density:
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        plt.scatter(x, y, c=z, s=50)
    else:
        plt.scatter(x, y, s=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x2 = np.linspace(np.min(x), np.max(x), 100)
    plt.plot(x2, reg.predict(x2.reshape(-1, 1)))
    plt.title(fr'$r^2$={score}, y = {np.round(reg.coef_[0,0], 3)}x + {np.round(reg.intercept_[0], 3)}' + title)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def plot_binom_helper2(x, y, xlabel, ylabel, ofile):
    reg = LinearRegression()
    reg.fit(x, y)
    score = np.round(reg.score(x, y), 3)
    yhat = reg.predict(x)
    print(reg.coef_, reg.intercept_)

    plt.scatter(yhat.flatten(), y.flatten(), s=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(fr'$r^2$={score}'
            f'\nMSE: {np.round(mean_squared_error(y, yhat), 3)}')
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def sc_structure_vs_sc_sample():
    # is sampling from bulk contact map the same as the individual structures
    # from the TICG model
    n_samples = 10
    for sample in [5, 15, 25, 34, 35]:
        dir = f'/home/erschultz/dataset_test_diag/samples/sample{sample}'
        odir = osp.join(dir, 'sc_contacts')
        y = np.load(osp.join(dir, 'y.npy')).astype(np.float32)
        y_proj_arr = []
        pca = PCA(n_components = 10)
        y_proj = pca.fit_transform(y).flatten().reshape(1, -1)

        save_sc_contacts(None, odir, triu = False, sparsify = False, overwrite = True,
                        fmt = 'txt', jobs = 10)

        avg_contacts = 0
        for i in range(n_samples):
            y_i = np.loadtxt(osp.join(odir, f'y_sc_{i}.txt'))
            proj = pca.transform(y_i)
            y_proj_arr.append(proj.flatten())

            contacts = int(np.sum(y_i) / 2)
            avg_contacts += contacts
            sparsity = np.round(np.count_nonzero(y_i) / len(y_i)**2 * 100, 2)
            plot_matrix(y_i, osp.join(odir, f'y_sc_{i}.png'), title = f'# contacts: {contacts}, sparsity: {sparsity}%', vmin = 0, vmax = 1)

        avg_contacts /= n_samples
        avg_contacts = int(avg_contacts)
        print(f'avg_contacts: {avg_contacts}')

        y = np.triu(y)
        np.fill_diagonal(y, 0)
        m = len(y)
        y = y[np.triu_indices(m)] # flatten
        y /= np.sum(y)
        rng = np.random.default_rng()
        for i in range(n_samples):
            y_i = np.zeros_like(y)
            for j in rng.choice(np.arange(len(y)), size = avg_contacts, p = y):
                y_i[j] += 1

            y_i = triu_to_full(y_i)
            proj = pca.transform(y_i)
            y_proj_arr.append(proj.flatten())

            contacts = int(np.sum(y_i) / 2)
            sparsity = np.round(np.count_nonzero(y_i) / len(y_i)**2 * 100, 2)
            plot_matrix(y_i, osp.join(odir, f'y_sample_{i}.png'), title = f'# contacts: {contacts}, sparsity: {sparsity}%', vmin = 0, vmax = 1)

        y_proj_arr = np.array(y_proj_arr)
        pca = PCA(n_components = 2)
        proj = pca.fit_transform(y_proj_arr)
        for i, vals in enumerate(proj):
            if i >= n_samples:
                shape = 'v'
                label = f'{i - n_samples}'
            else:
                shape = 'o'
                label = f'sc_{i}'
            plt.scatter(vals[0], vals[1], label = label, marker = shape)
            i += 1
        plt.legend()
        plt.savefig(osp.join(dir, 'sc vs sample.png'))
        plt.close()

def prep_data_for_cluster():
    dir = '/project2/depablo/erschultz/dataset_09_30_22/'
    odir = osp.join(dir, 'dataset_09_30_22_mini')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)
    idir = osp.join(dir, 'samples')
    if not osp.exists(idir):
        os.mkdir(odir, idir = 0o755)
    for id in os.listdir(idir):
        opath = osp.join(odir, id)
        if not osp.exists(opath):
            os.mkdir(opath, mode = 0o755)
        for file in ['y.npy', 's.npy', 'config.json', 'diag_chis_continuous.npy']:
            ifile = osp.join(idir, id, file)
            if osp.exists(ifile):
                ofile = osp.join(odir, id, file)
                copyfile(ifile, ofile)

def find_best_p_s():
    # is there a curve in training that matches experimental curve well?
    dir = '/home/erschultz/'
    data_dir = osp.join(dir, 'dataset_11_14_22/samples/sample1') # experimental data sample
    file = osp.join(data_dir, 'y.npy')
    y_exp = np.load(file)
    meanDist_ref = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')

    dir = '/home/erschultz/dataset_test_vbead' # simulated data dir
    # sort samples
    min_MSE = 1000
    best_sample = None
    best_meanDist = None
    for sample in range(4000):
        file = osp.join(dir, f'samples/sample{sample}', 'y.npy')
        if osp.exists(file):
            y = np.load(file)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            # meanDist = np.log(meanDist)
            mse = mean_squared_error(meanDist[:10], meanDist_ref[:10])
            if mse < min_MSE:
                min_MSE = mse
                best_sample = sample
                best_meanDist = meanDist
                print(sample, mse)

    print(best_sample, min_MSE)

    plt.plot(meanDist_ref[:10], label = 'ref')
    plt.plot(best_meanDist[:10], label = best_sample)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

def testGNNrank():
    '''
    check rank of GNN predicted E.
    tend to be (really) low rank unfortunately
    '''
    dir = '/home/erschultz'
    dataset = 'dataset_11_14_22'
    fig = plt.figure()
    for sample in [2217]:
        for GNN in [341]:
            print(GNN)
            file = osp.join(dir, dataset, f'samples/sample{sample}/GNN-{GNN}-S/k0/replicate1/resources/plaid_hat.txt')
            s = np.loadtxt(file)
            # print('rank', np.linalg.matrix_rank(s))
            # w, v = np.linalg.eig(s)
            # prcnt = np.abs(w) / np.sum(np.abs(w))

            pca = PCA()
            pca = pca.fit(s)
            prcnt = pca.explained_variance_ratio_

            print(prcnt[0:4])
            plt.plot(prcnt[:8], label = 'GNN')
            print(np.sum(prcnt[0:4]))
            # plot_top_PCs(s, verbose = True, count = 4)
            print()

        print('PCA')
        file = osp.join(dir, dataset, f'samples/sample{sample}/PCA-normalize-E/k8/replicate1/s.npy')
        s = np.load(file)
        print('rank', np.linalg.matrix_rank(s))
        # w, v = np.linalg.eig(s)
        # prcnt = np.abs(w) / np.sum(np.abs(w))

        pca = PCA()
        pca = pca.fit(s)
        prcnt = pca.explained_variance_ratio_
        plt.plot(prcnt[:8], label = 'PCA(k=8)')
        print(prcnt[0:4])
        print(np.sum(prcnt[0:4]))
        print()

    plt.ylabel('PC % Variance Explained', fontsize=16)
    plt.xlabel('PC', fontsize=16)
    plt.legend()
    plt.show()

def main():
    dataset = 'dataset_11_21_22'
    dir = f'/home/erschultz/{dataset}/samples/sample410'
    y = np.load(osp.join(dir, 'y.npy'))

    S = np.load(osp.join(dir, 's.npy'))
    S = (S + S.T)/2
    print(np.linalg.matrix_rank(S))
    E = s_to_E(S)
    with open(osp.join(dir, 'config.json')) as f:
        config = json.load(f)
    diag_chi_continuous = calculate_diag_chi_step(config)
    D = calculate_D(diag_chi_continuous)
    ED = calculate_net_energy(S, D)
    print('--'*5)

    val = 1
    S2 = S - val
    D2 = D + val
    ED2 = calculate_net_energy(S2, D2)
    print(np.allclose(ED, ED2))
    print('diff\n', ED - ED2)
    # plot_matrix(S, osp.join(dir, 'S.png'), title = 'S', cmap='blue-red')
    # plot_matrix(E, osp.join(dir, 'E.png'), title = 'E', cmap='blue-red')
    # plot_matrix(ED2, osp.join(dir, 'ED2.png'), title = 'ED2', cmap='blue-red')
    # plot_matrix(ED - ED2, osp.join(dir, 'diff.png'), title = 'ED - ED2', cmap='blue-red')


def main2():
    dataset = 'dataset_11_18_22'
    dir = f'/home/erschultz/{dataset}/samples/sample1462'
    SD = np.load(osp.join(dir, 'sd.npy'))
    np.fill_diagonal(SD, 0)

    dir = osp.join(dir, 'GNN-287-E/k0/replicate1')
    SD2 = np.load(osp.join(dir, 'resources/s.npy'))
    np.fill_diagonal(SD2, 0)

    vmin = np.min(SD)
    vmax = np.max(SD)

    # plot_matrix(SD, osp.join(dir, 'SD.png'), title = 'SD', cmap = 'blue-red', vmin=vmin, vmax=vmax)
    # plot_matrix(SD2, osp.join(dir, 'SD2.png'), title = 'SD2', cmap = 'blue-red', vmin=vmin, vmax=vmax)


    diff = SD - SD2
    # plot_matrix(diff, osp.join(dir, 'diff.png'), title = 'SD - SD_GNN', cmap = 'blue-red', vmin='min', vmax='max')

    print(mean_squared_error(SD, SD2))

    dir = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/287/dataset_11_18_22_sample1462-regular/sample1462-regular'
    SDgnn = np.loadtxt(osp.join(dir, 'energy.txt'))
    SD2gnn = np.loadtxt(osp.join(dir, 'energy_hat.txt'))
    diff = SD - SDgnn
    print(diff)
    plot_matrix(diff, osp.join(dir, 'diff.png'), title = 'SD - SDgnn', cmap = 'blue-red', vmin='min', vmax='max')

    diff2 = SD2 - SD2gnn
    print(diff2)
    plot_matrix(diff2, osp.join(dir, 'diff2.png'), title = 'SD - SDgnn', cmap = 'blue-red', vmin='min', vmax='max')
    print('mse', mean_squared_error(SDgnn, SD2gnn))

def plot_SCC_weights():
    dir = '/home/erschultz/dataset_11_14_22/samples/sample2203'
    y = np.load(osp.join(dir, 'y.npy'))
    pca_dir = osp.join(dir, 'PCA-normalize-E/k8/replicate1')
    y_pca = np.load(osp.join(pca_dir, 'y.npy'))

    K = 100
    scc, p_arr, w_arr = SCC().scc(y, y_pca, K = K, verbose = True)
    w_arr /= np.sum(w_arr)
    print('True', scc)
    plt.plot(w_arr, label = f'var_stabilized scc = {scc}')
    #
    # scc, p_arr, w_arr = SCC().scc(y, y_pca, K = K, var_stabilized = False, verbose = True)
    # w_arr /= np.sum(w_arr)
    # print('False', scc)
    # plt.plot(w_arr, label = f'scc = {scc}')

    w_arr = np.ones(len(w_arr))
    w_arr /= np.sum(w_arr)
    scc = np.sum(w_arr * p_arr)
    plt.plot(w_arr, label = f'mean = {scc}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # main()
    # main2()
    # temp()
    # prep_data_for_cluster()
    # find_best_p_s()
    # binom()
    # edit_argparse()
    # sc_nagano_to_dense()
    debugModel('ContactGNNEnergy')
    # testGNNrank()
    # plot_SCC_weights()
