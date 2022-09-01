import math
import multiprocessing
import os
import os.path as osp
import sys
import tarfile
import time
from shutil import copyfile, rmtree

import cooler
import hicrep.utils
import matplotlib.pyplot as plt
import numpy as np
import scHiCTools
import scipy.stats as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from core_test_train import core_test_train
from scipy import linalg
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import svd_flip
from utils.argparse_utils import (ArgparserConverter, finalize_opt,
                                  get_base_parser)
from utils.base_networks import AverageTo2d
from utils.energy_utils import s_to_E
from utils.load_utils import load_sc_contacts, save_sc_contacts
from utils.networks import get_model
from utils.plotting_utils import plot_matrix, plot_top_PCs
from utils.similarity_measures import SCC
from utils.utils import (DiagonalPreprocessing, calc_dist_strat_corr, crop,
                         print_time, triu_to_full)
from utils.xyz_utils import lammps_load


def test_num_workers():
    opt = argparseSetup() # get default args
    opt.data_folder = "/../../../project2/depablo/erschultz/dataset_04_18_21"
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
        if osp.isdir(type_path):
            for id in os.listdir(type_path):
                id_path = osp.join(type_path, id)
                if osp.isdir(id_path):
                    arg_file = osp.join(id_path, 'argparse.txt')
                    if osp.exists(arg_file):
                        with open(arg_file, 'r') as f:
                            lines = f.readlines()
                        weights = False
                        for i, line in enumerate(lines):

                            if line.startswith('preprocessing_norm'):
                                val = line[18:]
                                lines[i] = '--preprocessing_norm\n'+val
                                print(lines[i], id)
                        with open(arg_file, 'w') as f:
                            f.write("".join(lines))

def debugModel(model_type):
    parser = get_base_parser()
    AC = ArgparserConverter()
    opt = parser.parse_args()

    # dataset
    opt.data_folder = "/home/erschultz/dataset_test_diag1024"
    opt.scratch = '/home/erschultz/scratch'

    # architecture
    opt.m = 1024
    opt.y_preprocessing = None
    opt.split_percents=[0.8,0.2,0.0]
    # opt.split_percents = None
    # opt.split_sizes=[1, 2, 0]
    opt.split_sizes=None

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
        opt.loss = 'mse'
        opt.preprocessing_norm = 'instance'
        opt.message_passing='gat'
        opt.GNN_mode = True
        opt.output_mode = 'energy_sym'
        opt.encoder_hidden_sizes_list=None
        opt.update_hidden_sizes_list=None
        opt.hidden_sizes_list=[3,3]
        opt.act = 'relu'
        opt.inner_act = 'relu'
        opt.out_act = 'relu'
        opt.head_act = 'relu'
        opt.training_norm = 'instance'
        opt.use_edge_weights = False
        opt.use_edge_attr = True
        opt.transforms=AC.str2list('empty')
        opt.pre_transforms=AC.str2list('degree-contactdistance')
        opt.split_edges_for_feature_augmentation = False
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.log_preprocessing = None
        opt.head_architecture = 'bilinear_asym'
        opt.head_hidden_sizes_list = None
        opt.crop=[0,4]
        opt.m = 4
        opt.use_bias = True
        opt.num_heads = 2
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
        opt.head_architecture = 'bilinear_asym'
        opt.head_hidden_sizes_list = None
        opt.use_bias = True
        opt.use_bias = True
        opt.num_heads = 2
    elif model_type == 'MLP':
        opt.preprocessing_norm='instance'
        opt.random_split=True
        opt.hidden_sizes_list=AC.str2list('1000-'*6 + '20')
        opt.act='prelu'
        opt.out_act='prelu'
        opt.output_mode='diag_chi'
        opt.log_preprocessing='ln'
        opt.y_zero_diag_count=4
        # opt.training_norm='batch'
        opt.dropout=False
        opt.dropout_p=0.1
        # opt.crop=[20,1000]
        # opt.m = 980

    # hyperparameters
    opt.n_epochs = 80
    opt.lr = 1e-3
    opt.batch_size = 10
    opt.milestones = [20, 50]
    opt.gamma = 0.1

    # other
    opt.plot = True
    opt.plot_predictions = True
    opt.verbose = False
    opt.print_params = False
    opt.gpus = 0
    opt.delete_root = True
    opt.use_scratch = True
    opt.print_mod = 1
    # opt.id = 12
    # opt.resume_training = True

    opt = finalize_opt(opt, parser, False, debug = True)

    opt.model_type = model_type

    model = get_model(opt)

    core_test_train(model, opt)

    if opt.use_scratch:
        rmtree(opt.data_folder)

class ContactGNNEnergyTest(nn.Module):
    def __init__(self, m, head_hidden_sizes_list):
        super(ContactGNNEnergyTest, self).__init__()
        self.m = m

        ### Head Architecture ###
        head = []
        self.to2D = AverageTo2d(mode = 'outer')
        input_size = 4 # outer squares size
        for i, output_size in enumerate(head_hidden_sizes_list):
            head.append(nn.Linear(input_size, output_size, bias=False))
            input_size = output_size

        self.head = nn.Sequential(*head)
        with torch.no_grad():
            self.head[0].weight = nn.Parameter(torch.tensor([-1, 1, 1, 0],
                                                dtype = torch.float32))

    def forward(self, latent):
        _, output_size = latent.shape
        print('a', latent, latent.shape)
        latent = latent.reshape(-1, self.m, output_size)
        latent = latent.permute(0, 2, 1)
        print('b', latent, latent.shape)
        print(latent[:, :, 1], latent[:, :, 2])
        latent = self.to2D(latent)
        print('c', latent, latent.shape)
        print(latent[:, :, 1, 2])
        latent = latent.permute(0, 2, 3, 1)
        print('d', latent, latent.shape)
        out = self.head(latent)
        print('e', out, out.shape)

        return out

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

    # for i in [1,2,3,4,5,6,8,10,200]:
    #     print(f'\nk={i}')
    #     pca = PCA(n_components = i)
    #     s_transform = pca.fit_transform(s_sym)
    #     s_sym_i = pca.inverse_transform(s_transform)
    #
    #     pca = PCA(n_components = i)
    #     p_transform = pca.fit_transform(p)
    #     p_i = pca.inverse_transform(p_transform)
    #     loss = binom_nll(p_i[np.triu_indices(m)], y, n, 0)
    #     print(f'p_{i} loss', loss)
    #     plot_matrix(p_i, osp.join(dir, f'p_rank_{i}.png'), vmin = 'min',
    #                     vmax = 'max', title = f'p rank {i}')
    #
    #     # plotting
    #     plot_binom_helper(p_i[np.triu_indices(m)],
    #                         s_sym[np.triu_indices(m)],
    #                         r'$p_{ij}$',
    #                         r'$s_{ij}$',
    #                         osp.join(dir, f'p_rank_{i} vs s_sym.png'),
    #                         density = False)
    #     plot_binom_helper2(p_i,
    #                         s_sym,
    #                         r'$\hat{s}_{ij}$',
    #                         r'$s_{ij}$',
    #                         osp.join(dir, f'p_rank_{i}_s vs s_sym.png'))
    #     plot_binom_helper(p_i[np.triu_indices(m)],
    #                         s_sym_i[np.triu_indices(m)],
    #                         r'$p_{ij}$',
    #                         r'$s_{ij}$',
    #                         osp.join(dir, f'p_rank_{i} vs s_sym_{i}.png'),
    #                         density = False)
    #     plot_binom_helper2(p_i,
    #                         s_sym_i,
    #                         r'$\hat{s}_{ij}$',
    #                         r'$s_{ij}$',
    #                         osp.join(dir, f'p_rank_{i}_s vs s_sym_{i}.png'))
    #
    #     maxent_dir = osp.join(dir, f'PCA-normalize/k{i}/replicate1')
    #     s_file = osp.join(maxent_dir, 's.npy')
    #     if osp.exists(s_file):
    #         s_pca = np.load(s_file)
    #         s_sym_pca = (s_pca + s_pca.T)/2
    #         plot_matrix(s_sym_pca, osp.join(maxent_dir, f's_sym.png'), vmin = 'min',
    #                     vmax = 'max', title = r'$\hat{S}$', cmap='bluered')
    #         np.save(osp.join(maxent_dir, 's_sym.npy'), s_sym_pca)
    #
    #         # is p is the same as pca?
    #         plot_binom_helper(p_i[np.triu_indices(m)],
    #                             s_sym_pca[np.triu_indices(m)],
    #                             r'$p_{ij}$',
    #                             r'$sPCA_{ij}$',
    #                             osp.join(dir, f'p_rank_{i} vs s_sym_pca_{i}.png'))
    #         plot_binom_helper2(p_i,
    #                             s_sym_pca,
    #                             r'$\hat{S}_{ij}$',
    #                             r'$sPCA_{ij}$',
    #                             osp.join(dir, f'p_rank_{i}_s vs s_sym_pca_{i}.png'))
    #
    #     # optimal s_pca
    #     v = Vt[:i].T
    #     model = minimize(s_sym_pca_mse, np.zeros((i,i))[np.triu_indices(i)], args = (v, s_sym))
    #     print(model)
    #     s = v @ triu_to_full(model.x) @ v.T
    #     s_sym_pca_min = (s + s.T)/2
    #     optimal_pca_dict[i] = s_sym_pca_min
    #
    #     # is p is the same as optimal pca?
    #     plot_binom_helper(p_i[np.triu_indices(m)],
    #                         s_sym_pca_min[np.triu_indices(m)],
    #                         r'$p_{ij}$',
    #                         r'$sPCA_{ij}-min$',
    #                         osp.join(dir, f'p_rank_{i} vs s_sym_pca_{i}_min.png'))
    #     plot_binom_helper2(p_i,
    #                         s_sym_pca_min,
    #                         r'$\hat{s}_{ij}$',
    #                         r'$sPCA_{ij}-min$',
    #                         osp.join(dir, f'p_rank_{i} vs s_sym_pca_{i}_min.png'))
    #
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

def tar_samples():
    # iterates through results folder to find sample{i} folders
    # compresses sample{i} folder to .tar.gz
    results = '/home/erschultz/sequences_to_contact_maps/results'
    for dir in os.listdir(results):
        print(dir)
        for id in os.listdir(osp.join(results, dir)):
            id_path = osp.join(results, dir, id)
            if osp.isdir(id_path):
                print('', id)
                for file in os.listdir(id_path):
                    file_path = osp.join(id_path, file)
                    if file.startswith('sample') and osp.isdir(file_path):
                        tar_path = osp.join(id_path, file.split('.')[0] + '.tar.gz')
                        if not osp.exists(tar_path):
                            print('  ', file_path)
                            os.chdir(file_path)
                            with tarfile.open(tar_path, "x:gz") as tar:
                                for inner_file in os.listdir(file_path):
                                    tar.add(inner_file)
                        rmtree(file_path)

def main():
    # how to convert cooler to dense
    chrom = '10'
    resolution = 50000
    ubr = 500000
    h = 2
    K = int(ubr/resolution)+1
    times = 30

    dir1 = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017/samples/1CDS2.1'
    ifile = osp.join(dir1, 'adj.mcool')
    c1, binsize = hicrep.utils.readMcool(ifile, resolution)

    y1 = c1.matrix(balance=False).fetch(chrom)
    np.savetxt(osp.join(dir1, f'chrom{chrom}.txt'), y1)
    y1path = osp.join(dir1, f'chrom{chrom}.npy')
    np.save(y1path, y1[np.triu_indices(len(y1))])
    print(y1.shape)

    dir2 = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017/samples/1CDS2.2'
    ifile = osp.join(dir2, 'adj.mcool')
    c2, binsize = hicrep.utils.readMcool(ifile, resolution)

    y2 = c2.matrix(balance=False).fetch(chrom)
    np.savetxt(osp.join(dir2, f'chrom{chrom}.txt'), y2)
    y2path = osp.join(dir2, f'chrom{chrom}.npy')
    np.save(y2path, y2[np.triu_indices(len(y2))])


    t0 = time.time()
    for _ in range(times):
        val = hicrep.hicrepSCC(c1, c2, h, ubr, False,
                                [chrom])

    tf = time.time()
    # print(val)
    print_time(t0, tf, 'hicrepscc')
    print()

    t0 = time.time()
    scc = SCC()
    for _ in range(times):
        val1, p, w = scc.scc(y1, y2, h, K, True, verbose = True)
    tf = time.time()
    # print(val1)
    # print(p, p.shape,  w, w.shape)
    print_time(t0, tf, 'scc')

    t0 = time.time()
    scc = SCC()
    for _ in range(times):
        val2, p, w = scc.scc_file(y1path, y2path, h, K, True, verbose = True)
    tf = time.time()
    # print(val2)
    # print(p, p.shape,  w, w.shape)
    print_time(t0, tf, 'scc_file')


    t0 = time.time()
    scc = SCC()
    mapping = []
    for _ in range(times):
        mapping.append((y1path, y2path, h, K, True))
    with multiprocessing.Pool(15) as p:
        result = p.starmap(scc.scc_file, mapping)
    print(result)
    tf = time.time()
    print_time(t0, tf, 'scc_file_parallel')

    assert val1 == val2
    print(val1)

def main2():
    # scc comparison
    # using bulk hic data
    times = 100
    h = 1

    dir1 = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples/sample3'
    y1 = np.load(osp.join(dir1, 'y.npy'))
    y1 = uniform_filter(y1.astype(np.float64), 1+2*2, mode = 'constant')
    np.savetxt(osp.join(dir1, f'y.txt'), y1)

    dir2 = osp.join(dir1, 'PCA-normalize/k4/replicate1')
    y2 = np.load(osp.join(dir2, 'y.npy'))
    y2 = uniform_filter(y2.astype(np.float64), 1+2*2, mode = 'constant')

    np.savetxt(osp.join(dir2, f'y.txt'), y2)

    scc = SCC()
    t0 = time.time()
    for _ in range(times):
        val, p, w  = scc.scc(y1, y2, h, K = 10, var_stabilized = True, verbose = True)
    tf = time.time()
    print(val)
    print(p, p.shape)
    print(w, w.shape)
    print_time(t0, tf, 'scc')

def test_parallel():
    input_dir = '/home/erschultz/scratch/contact_diffusion_kNN4scc/iteration_0/sc_contacts'
    files = ['y_sc_6.npy', 'y_sc_11.npy', 'y_sc_0.npy', 'y_sc_8.npy', 'y_sc_9.npy', 'y_sc_10.npy', 'y_sc_7.npy', 'y_sc_4.npy', 'y_sc_5.npy', 'y_sc_3.npy', 'y_sc_2.npy', 'y_sc_1.npy']
    scc = SCC()
    N = len(files)
    D1 = np.zeros((N, N))
    mapping = []
    for i in range(0, N):
        ifile = osp.join(input_dir, files[i])
        for j in range(i, N):
            jfile = osp.join(input_dir, files[j])
            mapping.append((ifile, jfile, 1, 10, True))
            val = scc.scc_file(ifile, jfile, 1, 10, True)
            D1[i,j]  = val

    print(D1)
    D1 = np.triu(D1) + np.triu(D1, 1).T
    print(D1)
    D1 = 1 - D1
    print(D1, D1.shape)

    # with multiprocessing.Pool(15) as p:
    #     result = np.array(p.starmap(scc.scc_file, mapping))
    #
    # print('result')
    # print(1-result, result.shape)
    # print('triu')
    # print(D1[np.triu_indices(len(D1))])
    #
    # # make symmetric and convert to distance
    # D2 = 1 - triu_to_full(result)
    #
    # print('diff')
    # print(D2 - D1)

def prep_data_for_cluster():
    dir = '/home/erschultz/sequences_to_contact_maps/single_cell_nagano_2017'
    odir = osp.join(dir, 'samples_cluster')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)
    idir = osp.join(dir, 'samples')
    for id in os.listdir(idir):
        opath = osp.join(odir, id)
        if not osp.exists(opath):
            os.mkdir(opath, mode = 0o755)
        ifile = osp.join(idir, id, 'adj_500000.cool')
        if osp.exists(ifile):
            ofile = osp.join(odir, id, 'adj_500000.cool')
            copyfile(ifile, ofile)



if __name__ == '__main__':
    # main()
    # test_parallel()
    prep_data_for_cluster()
    # tar_samples()
    # binom()
    # edit_argparse()
    # sc_nagano_to_dense()
    # debugModel('MLP')
