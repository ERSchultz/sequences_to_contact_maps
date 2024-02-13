import json
import os
import os.path as osp
import time
from shutil import copyfile, rmtree

import cooler
import hicrep.utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import *
from pylib.utils.plotting_utils import BLUE_RED_CMAP, RED_CMAP, plot_matrix
from pylib.utils.similarity_measures import SCC
from pylib.utils.utils import print_time, triu_to_full
from scipy import linalg
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from core_test_train import core_test_train
from result_summary_plots import plot_top_PCs
from scripts.argparse_utils import (ArgparserConverter, finalize_opt,
                                    get_base_parser)
from scripts.load_utils import (get_converged_max_ent_folder, load_import_log,
                                load_L)
from scripts.neural_nets.dataset_classes import make_dataset
from scripts.neural_nets.losses import *
from scripts.neural_nets.networks import get_model
from scripts.neural_nets.utils import get_dataset
from scripts.utils import calc_dist_strat_corr, crop

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUV'

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
                            # if line.startswith('--relabel'):
                            #     assert lines.pop(i+1).strip() == 'false'
                            #     lines.pop(i)
                            # if line.startswith('--split_edges_for_feature_augmentation'):
                            #     if lines.pop(i+1).strip() == 'true':
                            #         split = True
                            #     else:
                            #         split = False
                            #     lines.pop(i)
                            #
                            #     i -= 2
                            #     assert lines[i].startswith('--pre_transforms')
                            #     i += 1
                            #     line = lines[i]
                            #
                            #     line_split = line.strip().split('-')
                            #     for j, val in enumerate(line_split):
                            #         if val.lower().startswith('degree'):
                            #             if split and 'split' not in val:
                            #                 line_split[j] = val + '_split'
                            #     line = '-'.join(line_split)+'\n'
                            #     lines[i] = line
                            # if line.startswith('--pre_transforms'):
                            #     i += 1
                            #     line = lines[i]
                            #     line_split = line.strip().split('-')
                            #     degree_count = 0
                            #     split = False
                            #     for j, val in enumerate(line_split):
                            #         if val.lower().startswith('degree'):
                            #             print(val)
                            #             degree_count += 1
                            #             if 'split' in val:
                            #                 print('h', val)
                            #                 split = True
                            #     if degree_count == 1 and split:
                            #         print('j')
                            #         line_split.insert(0, 'Degree')
                            #     line = '-'.join(line_split)+'\n'
                            #
                            #     lines[i] = line
                            if line.startswith('--use_scratch'):
                                lines.pop(i+1)
                                lines.pop(i)
                            i += 1
                        with open(arg_file, 'w') as f:
                            f.write("".join(lines))

def debugModel(model_type):
    parser = get_base_parser()
    AC = ArgparserConverter()
    opt = parser.parse_args()

    # dataset
    dir = "/home/erschultz"
    datasets = ['dataset_12_12_23_imr90']
    opt.data_folder = [osp.join(dir, d) for d in datasets]
    opt.scratch = '/home/erschultz/scratch'

    # architecture
    opt.m = 512
    opt.split_percents=[0.7,0.15,0.15]
    opt.split_sizes=None
    # opt.split_sizes=[1, 0, 0]
    # opt.split_percents = None
    opt.random_split=True

    if model_type == 'ContactGNNEnergy':
        opt.y_preprocessing = 'log_inf'
        opt.sweep_choices = None
        opt.rescale = 2
        opt.mean_filt = None
        opt.kr = False
        opt.keep_zero_edges = False
        opt.loss = 'scc_exp'
        opt.loss_k = 3
        opt.lambda1=5e-2
        opt.lambda2=1
        opt.preprocessing_norm = 'mean_fill'
        opt.message_passing = 'GAT'
        opt.GNN_mode = True
        opt.output_mode = 'energy_sym_diag'
        opt.output_preprocesing = 'none'
        opt.encoder_hidden_sizes_list=[30]
        opt.edge_encoder_hidden_sizes_list=[30]
        opt.update_hidden_sizes_list=[1000,1000,32]
        opt.hidden_sizes_list=[8,8,8]
        opt.gated = False
        opt.dropout = 0.0
        opt.act = 'leaky'
        opt.inner_act = 'relu'
        opt.out_act = 'prelu'
        opt.head_act = 'relu'
        opt.training_norm = None
        opt.use_node_features = False
        opt.k = 10
        opt.use_edge_weights = False
        opt.use_edge_attr = True
        # opt.transforms=AC.str2list('constant')
        opt.pre_transforms=['ContactDistance',
                            # 'ContactDistance_diagnorm',
                            # 'ContactDistance_corr',
                            # 'ContactDistance_corr_rank5',
                            'Constant',
                            # 'GridSize',
                            # 'GeneticDistance_norm',
                            # 'MeanContactDistance',
                            'MeanContactDistance_bonded']
        opt.mlp_model_id=None
        opt.sparsify_threshold = None
        opt.sparsify_threshold_upper = None
        opt.log_preprocessing = None
        opt.head_architecture = 'bilinear'
        opt.input_L_to_D = False
        opt.input_L_to_D_mode = 'subtract'
        opt.head_architecture_2 = f'fc-fill_{opt.m}'
        opt.head_hidden_sizes_list = [1000, 1000,1000]
        # opt.crop = [128, 256]
        opt.plaid_score_cutoff = None

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
        opt.encoder_hidden_sizes_list=None
        opt.edge_encoder_hidden_sizes_list=[8]
        opt.update_hidden_sizes_list=[100,100,32]
        opt.hidden_sizes_list=[8]
        opt.act = 'prelu'
        opt.inner_act = 'relu'
        opt.out_act = 'relu'
        opt.head_act = 'relu'
        opt.use_edge_attr = True
        # opt.transforms=AC.str2list('constant')
        opt.pre_transforms=AC.str2list('ContactDistance-GeneticDistance_norm-AdjPCs_8')
        opt.k=8
        opt.head_architecture = 'dconv-bilinear-triu'
        opt.head_architecture_2 = f'dconv-fc-fill_{opt.m}'
        opt.head_hidden_sizes_list = [1000,1000,1000]
        opt.crop = None

        opt.use_bias = True
        opt.num_heads = 6
        opt.concat_heads = True

    # hyperparameters
    opt.n_epochs = 1
    opt.lr = 1e-3
    opt.weight_decay = 1e-5
    opt.w_reg = None; opt.reg_lambda = 10
    opt.batch_size = 4
    opt.gamma = 0.1

    # other
    opt.pretrain_id = None
    opt.plot = False
    opt.plot_predictions = False
    opt.verbose = True
    opt.print_params = True
    opt.gpus = 1
    # opt.delete_root = True
    opt.num_workers = 2
    opt.use_scratch_parallel = False

    opt = finalize_opt(opt, parser, False, debug = True)
    opt.model_type = model_type
    model = get_model(opt)
    core_test_train(model, opt)

    if opt.move_data_to_scratch:
        rmtree(opt.data_folder)

def binom():
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

def testGNNrank(dataset, GNN_ID):
    '''
    check rank of GNN predicted E.
    tend to be (really) low rank unfortunately
    '''
    dir = f'/home/erschultz/{dataset}/samples'
    samples = list(range(201, 211))

    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(len(samples)) % cmap.N
    colors = cmap(ind)
    pca = PCA()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.get_yaxis().set_visible(False)
    for i, sample in enumerate(samples):
        print(sample)
        s_dir = osp.join(dir, f'sample{sample}')
        S_gnn = np.load(osp.join(s_dir, f'GNN-{GNN_ID}-E/k0/replicate1/resources/s.npy'))

        ax2.plot(np.NaN, np.NaN, label = sample, color = colors[i])


        print('GNN')
        pca = pca.fit(S_gnn)
        prcnt = pca.explained_variance_ratio_
        print(prcnt[0:4])
        ax.plot(prcnt[:8], label = 'GNN', color = colors[i], ls = '--')
        print(np.sum(prcnt[0:4]))
        print()

        pca_dir = osp.join(s_dir, f'PCA-normalize-E/k8/replicate1')
        L = load_L(pca_dir)
        with open(osp.join(pca_dir, 'iteration13/config.json'), 'r') as f:
            config = json.load(f)
        diag_chis_continuous = calculate_diag_chi_step(config)
        D = calculate_D(diag_chis_continuous)
        S = calculate_S(L, D)
        pca = pca.fit(S)
        prcnt = pca.explained_variance_ratio_

        print('Sim')
        print(prcnt[0:4])
        ax.plot(prcnt[:8], label = 'Sim', color = colors[i])
        print(np.sum(prcnt[0:4]))
        print()

    ax.set_ylabel('PC % Variance Explained', fontsize=16)
    ax.set_xlabel('PC', fontsize=16)
    ax2.legend(loc='upper right')
    ax.legend(loc='lower right')
    plt.title(dataset)
    plt.show()

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

def gnn_meanDist_s(GNN_ID, sample):
    dir = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy'
    dataset = 'dataset_04_28_23'
    dir = osp.join(dir, f'{GNN_ID}/{dataset}_sample{sample}/sample{sample}')
    assert osp.exists(dir), f'{dir} does not exist'
    e_gt = np.loadtxt(osp.join(dir, 'energy.txt'))
    e_hat = np.loadtxt(osp.join(dir, 'energy_hat.txt'))

    ref = np.load(f'/home/erschultz/{dataset}/samples/sample{sample.split("-")[0]}/S.npy')
    print(ref[0,:10])
    ref_mean = np.mean(ref)
    ref_center = ref - ref_mean
    ref_max = np.max(np.abs(ref_center))
    ref_center_norm = ref_center / ref_max
    ref_center_norm_log = np.sign(ref_center_norm) * np.log(np.abs(ref_center_norm)+1)
    # assert np.allclose(ref_center_norm_log, e_gt, rtol=1e-03, atol=1e-03),
            # f'diff {ref_center_norm_log - e_gt}'

    e_orig_gt = e_gt
    e_orig_hat = e_hat
    e_orig_gt = np.multiply(np.sign(e_gt), np.exp(np.abs(e_gt)) - 1)
    e_orig_hat = np.multiply(np.sign(e_hat), np.exp(np.abs(e_hat)) - 1)
    # e_orig_gt *= ref_max
    # e_orig_gt += ref_mean
    # e_orig_hat *= ref_max
    # e_orig_hat += ref_mean
    diff = e_orig_gt - e_orig_hat
    print(e_orig_gt[0,:10])
    rmse = mean_squared_error(e_orig_gt, e_orig_hat, squared = False)
    print(rmse)

    # assert np.allclose(ref, e_orig_gt, rtol=1e-01, atol=1e-01),
        # f'diff {ref - e_orig_gt} rmse {mean_squared_error(ref, e_orig_gt, squared = False)}'

    fig, axes = plt.subplots(1, 4,
                                    gridspec_kw={'width_ratios':[1,1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    fig.suptitle(f'Sample {sample}', fontsize = 16)
    arr = np.array([e_orig_gt, e_orig_hat])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    labels = ['Ground Truth', f'GNN Estimate\nRMSE={rmse:.2f}', 'Different (L-R)']
    matrices = [e_orig_gt, e_orig_hat, diff]
    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        ax = axes[i]
        if i == len(matrices)-1:
            s = sns.heatmap(matrix, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = BLUE_RED_CMAP,
                            ax = ax, cbar_ax = axes[-1])
        else:
            s = sns.heatmap(matrix, linewidth = 0, vmin = vmin, vmax = vmax,
                            cmap = BLUE_RED_CMAP,
                            ax = ax, cbar = False)
        s.set_title(label, fontsize = 16)
        s.set_yticks([])
    plt.tight_layout()
    plt.savefig(osp.join(dir, 'energy_orig_comparison.png'))
    plt.close()


    for e, label in zip([e_gt, e_hat],['Ground Truth', 'GNN']):
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(e, 'freq')
        plt.plot(meanDist, label = label)
    plt.legend()
    plt.xscale('log')
    plt.ylabel('Mean', fontsize=16)
    plt.xlabel('Off-diagonal Index', fontsize=16)
    plt.savefig(osp.join(dir, 'meanDist_S.png'))
    plt.close()

def test_center_norm_log():
    ref = np.random.rand(2,4)
    ref *= 6
    ref += 2
    print('ref', ref)
    ref_mean = np.mean(ref)
    print(ref_mean)
    ref_center = ref - ref_mean
    print('ref_center', ref_center)
    ref_max = np.max(np.abs(ref_center))
    ref_center_norm = ref_center / ref_max
    print('ref_center_norm', ref_center_norm)
    ref_center_norm_log = np.sign(ref_center_norm) * np.log(np.abs(ref_center_norm)+1)
    print('ref_center_norm_log', ref_center_norm_log)

    ref2 = ref_center_norm_log.copy()
    ref2 = np.round(ref2, 3)

    ref2 = np.multiply(np.sign(ref2), np.exp(np.abs(ref2)) - 1)
    ref2 *= ref_max
    # print(ref2)
    ref2 += ref_mean
    print('final', ref2)

def temp():
    a = np.array([0.45440109, 0.27298974, -0.1844925])
    a2 = a**2
    print(a2)
    print(np.sum(a2))

def test_loss():
    N=2
    m=3
    k=2
    torch.manual_seed(12)
    vecs = torch.randn((N, k, m), dtype=torch.float32)
    e = torch.randn((N, m, m), dtype=torch.float32)
    e = e + torch.transpose(e, 1, 2)
    ehat = torch.randn((N, m, m), dtype=torch.float32)
    ehat = ehat + torch.transpose(ehat, 1, 2)

    metric6 = MSE_plaid(len(e))
    # mse6 = metric6(e, ehat)
    # print(mse6)

    metric7 = MSE_plaid_eig()
    # mse7 = metric7(e, ehat, vecs)
    # print('mse7', mse7)


    metricC = Combined_Loss([metric6, metric7], [1, 0.1], [None, None])
    # mseC = metricC(e, ehat, [None, vecs])
    # print(mseC)

    metric8 = MSE_EXP_NORM()
    mse8 = metric8(e, ehat)
    print('mse8', mse8)


def S_to_Y():
    dir = '/home/erschultz/dataset_12_06_23/samples/sample1'
    grid_root = 'optimize_grid_b_200_v_8_spheroid_1.5'
    y_bonded = np.load(osp.join(dir, grid_root, 'y.npy'))
    y_bonded /= np.mean(y_bonded.diagonal())

    scc = SCC(h=1, K=100)

    me_dir = osp.join(dir, f'{grid_root}-max_ent10')
    S_me = np.load(osp.join(me_dir, 'iteration30/S.npy'))

    gnn_dir = osp.join(dir, f'{grid_root}-GNN629')
    odir = osp.join(gnn_dir, 'test')
    if not osp.exists(odir):
        os.mkdir(odir)

    y = np.load(osp.join(gnn_dir, 'y.npy'))
    plot_matrix(y, osp.join(odir, 'y.png'), vmax = 'mean')

    S = np.load(osp.join(gnn_dir, 'S.npy'))
    print('S', scc.scc(S, S_me))
    print('exp(S)', scc.scc(np.exp(-S), np.exp(-S_me)))
    print('log(S)', scc.scc(np.sign(S) * np.log(np.abs(S) + 1),
                            np.sign(S_me) * np.log(np.abs(S_me) + 1)))

    y_S = np.exp(-S)
    # y_S /= np.mean(y_S.diagonal())
    print('y_S', scc.scc(y, y_S))
    plot_matrix(y_S, osp.join(odir, 'y_S.png'), vmax = 'mean')
    #
    # y_add = y_bonded + y_S
    # y_add /= np.mean(y_add.diagonal())
    # print('add', scc.scc(y, y_add))
    # plot_matrix(y_add, osp.join(odir, 'y_add.png'), vmax = 'mean')
    #
    # y_mul = y_bonded * y_S
    # y_mul /= np.mean(y_mul.diagonal())
    # print('mul', scc.scc(y, y_mul))
    # plot_matrix(y_mul, osp.join(odir, 'y_mul.png'), vmax = 'mean')

if __name__ == '__main__':
    # temp()
    # test_loss()
    # binom()
    # edit_argparse()
    debugModel('ContactGNNEnergy')
    # gnn_meanDist_s(434, '981')
    # test_center_norm_log()
    # testGNNrank('dataset_02_04_23', 378)
    # plot_SCC_weights()
    # S_to_Y()
