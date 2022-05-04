import os
import os.path as osp
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from core_test_train import core_test_train
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from utils.argparse_utils import finalize_opt, get_base_parser, str2list
from utils.base_networks import AverageTo2d
from utils.energy_utils import s_to_E
from utils.load_utils import load_sc_contacts
from utils.networks import get_model
from utils.plotting_utils import plot_matrix, plot_top_PCs
from utils.utils import (DiagonalPreprocessing, calc_dist_strat_corr, crop,
                         print_time, triu_to_full)
from utils.xyz_utils import lammps_load


def test_num_workers():
    opt = argparseSetup() # get default args
    opt.data_folder = "/../../../project2/depablo/erschultz/dataset_04_18_21"
    opt.cuda = True
    opt.device = torch.device('cuda')
    opt.y_preprocessing = 'diag'
    opt.y_norm = 'batch'
    dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
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

                            if line == '--y_log_transform\n' and lines[i+1] == 'true\n':
                                lines[i+1] = '10\n'
                                print(lines[i+1], id)
                        with open(arg_file, 'w') as f:
                            f.write("".join(lines))

def debugModel(model_type):
    parser = get_base_parser()
    opt = parser.parse_args()

    # dataset
    opt.data_folder = "/home/erschultz/dataset_test2"
    opt.scratch = '/home/erschultz/scratch'

    # architecture
    opt.m = 1024
    opt.y_preprocessing = 'diag_batch'
    # opt.split_percents=[0.6666,0.3333,0.0]
    opt.split_percents = None
    opt.split_sizes=[1, 2, 0]
    # opt.split_counts=None

    if model_type == 'Akita':
        opt.kernel_w_list=str2list('5-5-5')
        opt.hidden_sizes_list=str2list('4-6-8')
        opt.dilation_list_trunk=str2list('2-4-8-16')
        opt.bottleneck=4
        opt.dilation_list_head=str2list('2-4-8-16')
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
        opt.kernel_w_list=str2list('5-5-5')
        opt.hidden_sizes_list=str2list('32-64-128')
        opt.dilation_list=str2list('2-4-8-16-32-64-128-256-512')
    elif model_type == 'ContactGNN':
        opt.GNN_mode = True
        opt.output_mode = 'sequence'
        opt.loss = 'BCE'
        opt.y_norm = None
        opt.message_passing='SignedConv'
        opt.hidden_sizes_list=str2list('16-2')
        opt.out_act = None
        opt.use_node_features = False
        opt.use_edge_weights = False
        opt.transforms=str2list('none')
        opt.pre_transforms=str2list('degree')
        opt.split_neg_pos_edges_for_feature_augmentation = True
        opt.top_k = None
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.relabel_11_to_00 = False
        opt.y_log_transform = 'True'
        opt.head_architecture = 'fc'
        opt.head_hidden_sizes_list = [2]
        # opt.crop=[50,100]
        # opt.m = 50
        # opt.use_bias = False
    elif model_type == 'ContactGNNEnergy':
        opt.loss = 'mse'
        opt.y_norm = None
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
        opt.transforms=str2list('empty')
        opt.pre_transforms=str2list('degree-contactdistance')
        opt.split_edges_for_feature_augmentation = False
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.y_log_transform = 'ln'
        opt.head_architecture = 'bilinear_asym'
        opt.head_hidden_sizes_list = None
        opt.crop=[0,4]
        opt.m = 4
        opt.use_bias = True
        opt.num_heads = 2
        opt.concat_heads = True

    # hyperparameters
    opt.n_epochs = 3
    opt.lr = 1e-4
    opt.batch_size = 1
    opt.milestones = None
    opt.gamma = 0.1

    # other
    opt.plot = True
    opt.plot_predictions = True
    opt.verbose = True
    opt.print_params = False
    opt.gpus = 0
    opt.delete_root = True
    opt.use_scratch = False
    opt.print_mod = 1
    # opt.id = 12
    # opt.resume_training = True

    opt = finalize_opt(opt, parser, False, debug = True)

    opt.model_type = model_type

    model = get_model(opt)

    core_test_train(model, opt)

def test_argpartition(k):
    path = 'dataset_04_18_21\\samples\\sample1'
    y = np.load(osp.join(path, 'y.npy'))
    k = len(y) - k
    print(y, y.shape)
    print(y[3])
    z = np.partition(y, k)
    z =z[:, k:]
    print(z[3], z[3].shape)

    miny = np.min(y, axis = 1)
    minz = np.min(z, axis = 1)
    print(miny)
    print(minz)

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
            self.head[0].weight = nn.Parameter(torch.tensor([-1, 1, 1, 0], dtype = torch.float32))

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

def testEnergy():
    dir = "C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps"
    ofile = osp.join(dir, "results/test/energy")
    if not osp.exists(ofile):
        os.mkdir(ofile, mode = 0o755)

    m = 1024

    model = ContactGNNEnergyTest(m, [1])
    tot_pars = 0
    for k,p in model.named_parameters():
        tot_pars += p.numel()
        print(k, p.numel(), p.shape)
        print(p)
    print('Total parameters: {}'.format(tot_pars))

    x = np.load(osp.join(dir, "dataset_04_18_21/samples/sample40/x.npy"))[:m]
    # x = np.array([[0,0], [0,1], [1,0], [1,1]])[:m]

    chi = np.load(osp.join(dir, 'dataset_04_18_21/chis.npy'))
    print(chi)
    energy = x @ chi @ x.T
    print(energy)

    x = torch.tensor(x, dtype = torch.float32)

    z = np.load(osp.join(dir, "results/ContactGNN/159/sample40/z.npy"))[:m]
    z = torch.tensor(z, dtype = torch.float32)

    energy_hat = model(z)

    energy_hat = energy_hat.cpu().detach().numpy()
    energy_hat = energy_hat.reshape((m,m))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'white'),
                                              (1,    'blue')], N=126)
    # not a contat map but using this function anyways
    v_max = np.max(energy)
    v_min = np.min(energy)
    plot_matrix(energy_hat, osp.join(ofile, 'energy_hat.png'), vmin = v_min, vmax = v_max, cmap = cmap, title = r'$\hat{S}$')
    plot_matrix(energy, osp.join(ofile, 'energy.png'), vmin = v_min, vmax = v_max, cmap = cmap, title = r'$S$')

def test_lammps_load():
    file_path = '/home/erschultz/sequences_to_contact_maps/traj.dump.lammpstrj'
    xyz = lammps_load(file_path)

def binom():
    if False:
        # toy data
        p = np.ones((m,m))
        for i in range(m):
            for j in range(i):
                p[i,j] = (i+j)/(2*m)
                p[j,i] = (i+j)/(2*m)
        print(p)
        p = p[np.triu_indices(m)]
        print('p', p)

        y = np.zeros_like(p)
        n=1000
        for i in range(len(p)):
            y[i] = ss.binom.rvs(n, p[i], size = 1)
        print('y', y)

    dir = '/home/erschultz/sequences_to_contact_maps/dataset_01_17_22/samples/sample1'
    dir = '/home/erschultz/dataset_test/samples/sample10'
    y = np.load(osp.join(dir, 'y_diag.npy'))
    m = len(y)
    e = np.load(osp.join(dir, 'e.npy'))
    # s = np.load(osp.join(dir, 's.npy'))
    y = y[np.triu_indices(m)]
    n = np.max(y)
    print('y', y, y.shape)

    p = y/n
    loss = binom_nll(p, y, n, 100)
    print('p loss', loss)
    p = triu_to_full(p)
    print('p', p, p.shape)
    plot_matrix(p, osp.join(dir, 'p.png'), title = None, vmin = 'min', vmax = 'max')
    # plot_top_PCs(p, 'p', dir, sys.stdout, count = 4, plot = True, verbose = True)


    for i in [1,2,3,4,5,10,200]:
        # get y top i PCs
        pca = PCA(n_components = i)
        p_transform = pca.fit_transform(p)
        p_i = pca.inverse_transform(p_transform)
        if i == 1:
            p_1 = p_i
        loss = binom_nll(p_i[np.triu_indices(m)], y, n, 0)
        print(f'p_{i} loss', loss)
        plot_matrix(p_i, osp.join(dir, f'p_rank_{i}.png'), vmin = 'min', vmax = 'max', title = f'p rank {i}')

        # VVT = pca.components_.T @ pca.components_
        # print(pca.components_.shape)
        # plot_matrix(VVT, osp.join(dir, f'VVT_{i}.png'), vmin = 'min', vmax = 'max', title = f'VVT {i}')
        # x = VVT[np.triu_indices(m)]
        # s = np.load(osp.join(dir, 'PCA/k1/replicate1/s.npy'))
        # y = s_to_E(s)[np.triu_indices(m)]
        #
        # reg = LinearRegression()
        # reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        # score = np.round(reg.score(x.reshape(-1, 1), y.reshape(-1, 1)), 3)
        # yhat = triu_to_full(reg.predict(x.reshape(-1, 1)).reshape(-1))
        # plot_matrix(yhat, osp.join(dir, f'VVT_{i}_e_pca_hat.png'), vmin = 'min', vmax = 'max', title = f'VVT_{i} pca ehat')
        # np.save(osp.join(dir, f'VVT_{i}_e_pca_hat.npy'), yhat)
        #
        # # density plot
        # xy = np.vstack([x,y])
        # z = gaussian_kde(xy)(xy)
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]
        #
        # plt.scatter(x, y, c=z, s=50)
        # plt.ylabel(r'$e_{ij}$')
        # plt.xlabel(r'$p_{ij}$')
        # x2 = np.linspace(np.min(x), np.max(x), 100)
        # plt.plot(x2, reg.predict(x2.reshape(-1, 1)))
        # plt.title(fr'$r^2$={score}, e = {np.round(reg.coef_[0,0], 3)}p + {np.round(reg.intercept_[0], 3)}')
        # plt.tight_layout()
        # plt.savefig(osp.join(dir, f'VVT_{i} vs e_pca density.png'))
        # plt.close()

        # plotting
        x = p_i[np.triu_indices(m)]
        y = e[np.triu_indices(m)]

        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        score = np.round(reg.score(x.reshape(-1, 1), y.reshape(-1, 1)), 3)
        yhat = triu_to_full(reg.predict(x.reshape(-1, 1)).reshape(-1))
        plot_matrix(yhat, osp.join(dir, f'p_rank_{i}_ehat.png'), vmin = 'min', vmax = 'max', title = f'p rank {i} ehat')
        np.save(osp.join(dir, f'p_rank_{i}_ehat.npy'), yhat)

        plt.scatter(x, y, s=50)
        plt.ylabel(r'$e_{ij}$')
        plt.xlabel(r'$p_{ij}$')
        x2 = np.linspace(np.min(x), np.max(x), 100)
        plt.plot(x2, reg.predict(x2.reshape(-1, 1)))
        plt.title(fr'$r^2$={score}, e = {np.round(reg.coef_[0,0], 3)}p + {np.round(reg.intercept_[0], 3)}')
        plt.savefig(osp.join(dir, f'p_rank_{i} vs e.png'))
        plt.close()

        # density plot
        # xy = np.vstack([x,y])
        # z = gaussian_kde(xy)(xy)
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]
        #
        # plt.scatter(x, y, c=z, s=50)
        # plt.ylabel(r'$e_{ij}$')
        # plt.xlabel(r'$p_{ij}$')
        # plt.plot(x2, reg.predict(x2.reshape(-1, 1)))
        # plt.title(fr'$r^2$={score}, e = {np.round(reg.coef_[0,0], 3)}p + {np.round(reg.intercept_[0], 3)}')
        # plt.tight_layout()
        # plt.savefig(osp.join(dir, f'p_rank_{i} vs e density.png'))
        # plt.close()

    x = p_1[np.triu_indices(m)]
    s = np.load(osp.join(dir, 'PCA/k1/replicate1/s.npy'))
    y = s_to_E(s)[np.triu_indices(m)]

    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    score = np.round(reg.score(x.reshape(-1, 1), y.reshape(-1, 1)), 3)
    yhat = triu_to_full(reg.predict(x.reshape(-1, 1)).reshape(-1))
    plot_matrix(yhat, osp.join(dir, f'p_rank_1_e_pca_hat.png'), vmin = 'min', vmax = 'max', title = f'p rank 1 pca ehat')
    np.save(osp.join(dir, f'p_rank_1_e_pca_hat.npy'), yhat)

    # density plot
    # xy = np.vstack([x,y])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]
    #
    # plt.scatter(x, y, c=z, s=50)
    # plt.ylabel(r'$e_{ij}$')
    # plt.xlabel(r'$p_{ij}$')
    # x2 = np.linspace(np.min(x), np.max(x), 100)
    # plt.plot(x2, reg.predict(x2.reshape(-1, 1)))
    # plt.title(fr'$r^2$={score}, e = {np.round(reg.coef_[0,0], 3)}p + {np.round(reg.intercept_[0], 3)}')
    # plt.tight_layout()
    # plt.savefig(osp.join(dir, f'p_rank_1 vs e_pca density.png'))
    # plt.close()



    print('-'*30)
    lmbda = 100
    model = minimize(binom_nll, np.ones_like(y) / 2, args = (y, n, lmbda),
                    bounds = [(0,1)]*len(y))
    print(model)

def binom_nll(p, y, n, lmbda = 100):
    print('\t', p)
    ll = 0
    l = 0
    for i in range(len(p)):
        # print(p[i], y[i])
        left = p[i]**y[i]
        right = (1 - p[i])**(n - y[i])
        likelihood = left*right
        l += likelihood
        # ll += np.log(likelihood)
    print('\t', l)
    nll = -1 * np.log(l/(1+l))
    print('\t', nll)


    p_full = triu_to_full(p)
    reg = lmbda * np.linalg.norm(p_full, 'nuc')
    print('\t', likelihood, reg)

    return likelihood + reg


def plot_e_from_s():
    dir = '/home/erschultz/dataset_test/samples/sample10/'
    maxent_dir = osp.join(dir, 'PCA/k1/replicate1')
    x = np.load(osp.join(maxent_dir, 'resources/x.npy'))

    plt.plot()




if __name__ == '__main__':
    # plot_e_from_s()
    binom()
    # edit_argparse()
    # debugModel('ContactGNNEnergy')
    # test_lammps_load()
    # plot_fixed()
    # test_argpartition(10)
    # downsampling_test()
    # testEnergy()
