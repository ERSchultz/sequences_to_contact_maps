import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core_test_train import core_test_train
from utils.argparse_utils import finalize_opt, get_base_parser, str2list
from utils.base_networks import AverageTo2d
from utils.load_utils import load_sc_contacts
from utils.networks import get_model
from utils.plotting_utils import plot_matrix
from utils.utils import (calc_dist_strat_corr, crop, diagonal_preprocessing,
                         genomic_distance_statistics, print_time, triu_to_full)
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
                        print(arg_file)
                        with open(arg_file, 'r') as f:
                            lines = f.readlines()
                        for i, line in enumerate(lines):
                            if line == '--n\n':
                                lines[i] = '--m\n'
                                break
                        with open(arg_file, 'w') as f:
                            f.write("".join(lines))

def debugModel(model_type):
    parser = get_base_parser()
    opt = parser.parse_args()

    # dataset
    opt.data_folder = "/home/erschultz/dataset_test"
    opt.scratch = '/home/erschultz/scratch'
    opt.use_scratch=True

    # architecture
    opt.k = 2
    opt.crop = None
    opt.m = 1024
    opt.y_preprocessing = 'diag'
    # opt.split_percents=[0.6666,0.3333,0.0]
    opt.split_percents = None
    opt.split_sizes=[8, 2, 0]
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
    elif model_type == 'GNNAutoencoder':
        opt.GNN_mode = True
        opt.autoencoder_mode=True
        opt.hidden_sizes_list=str2list('8-2')
        opt.out_act = 'relu'
        opt.use_node_features = False
        opt.pre_transforms=str2list('constant')
        opt.top_k = 100
    elif model_type == 'GNNAutoencoder2':
        opt.GNN_mode = True
        opt.autoencoder_mode=True
        opt.hidden_sizes_list=str2list('12-4')
        opt.head_hidden_sizes_list=str2list('200-25')
        opt.out_act = 'relu'
        opt.head_architecture ='FCAutoencoder'
        opt.use_node_features = False
        opt.transforms=str2list('constant')
        opt.pre_transforms=str2list('weighted_LDP')
        opt.top_k = 500
        opt.parameter_sharing = True
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
        opt.y_log_transform = True
        opt.head_architecture = 'fc'
        opt.head_hidden_sizes_list = [2]
        # opt.crop=[50,100]
        # opt.m = 50
        # opt.use_bias = False
    elif model_type == 'ContactGNNEnergy':
        opt.loss = 'mse'
        opt.y_norm = None
        opt.message_passing='gcn'
        opt.GNN_mode = True
        opt.output_mode = 'energy'
        opt.encoder_hidden_sizes_list=[100, 100, 8]
        opt.update_hidden_sizes_list=[100, 100, 16]
        opt.hidden_sizes_list=[8, 8, 2]
        opt.act = 'relu'
        opt.inner_act = 'prelu'
        opt.out_act = 'prelu'
        opt.head_act = 'prelu'
        opt.use_node_features = False
        opt.use_edge_weights = True
        opt.transforms=str2list('none')
        opt.pre_transforms=str2list('degree')
        opt.split_neg_pos_edges_for_feature_augmentation = False
        opt.top_k = None
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.relabel_11_to_00 = False
        opt.y_log_transform = False
        opt.head_architecture = 'concat'
        opt.head_hidden_sizes_list = [100,100,1]
        opt.crop=[0,50]
        opt.m = 50
        # opt.use_bias = False
    elif model_type == 'SequenceFCAutoencoder':
        opt.output_mode = 'sequence'
        opt.autoencoder_mode = True
        opt.out_act = None
        opt.hidden_sizes_list=str2list('1024-1024-128')
        opt.parameter_sharing = False
    elif model_type == 'SequenceConvAutoencoder':
        opt.output_mode = 'sequence'
        opt.autoencoder_mode = True
        opt.out_act = None
        opt.hidden_sizes_list=str2list('4-8-12-128')

    # hyperparameters
    opt.n_epochs = 5
    opt.lr = 1e-3
    opt.batch_size = 1
    opt.milestones = None
    opt.gamma = 0.1

    # other
    opt.plot = True
    opt.plot_predictions = True
    opt.verbose = False
    opt.print_params = False
    opt.gpus = 0
    opt.delete_root = False
    opt.use_scratch = False

    opt = finalize_opt(opt, parser, False)

    opt.model_type = model_type

    model = get_model(opt)

    # opt.model_type = 'test'
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

def plot_fixed():
    samples = [11, 12]
    for i in samples:
        for j in samples:
            if i >= j:
                continue
            y1 = np.load('dataset_fixed/samples/sample{}/y.npy'.format(i))
            y2 = np.load('dataset_fixed/samples/sample{}/y.npy'.format(j))

            x1 = np.load('dataset_fixed/samples/sample{}/x.npy'.format(i))
            x2 = np.load('dataset_fixed/samples/sample{}/x.npy'.format(j))
            assert np.array_equal(x1, x2)

            overall_corr, corr_arr = calc_dist_strat_corr(y1, y2, mode = 'pearson')
            avg = np.nanmean(corr_arr)
            title = f'''Overall Pearson R: {np.round(overall_corr, 3)}\n
                    Average Dist Pearson R: {np.round(avg, 3)}'''

            plt.plot(np.arange(1022), corr_arr, color = 'black')
            plt.ylim(-0.5, 1)
            plt.xlabel('Distance', fontsize = 16)
            plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)
            plt.title(title, fontsize = 16)

            plt.tight_layout()
            plt.savefig('dataset_fixed/samples/distance_pearson_i{}j{}.png'.format(i, j))
            plt.close()

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

def main():
    dir = 'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps/dataset_01_15_22/samples/sample1230'
    y_diag = np.load(osp.join(dir, 'y_diag.npy'))
    y = np.load(osp.join(dir, 'y.npy'))
    # plot_matrix(y_diag, osp.join(dir, 'y_diag_blue_red.png'), vmax = 2, cmap = 'blue-red')
    # plot_matrix(y, osp.join(dir, 'y_120.png'), vmax = 120)

    y_diag[[np.logical_and(y_diag < 1.5, y_diag > 0.5)]] = 1
    print(y_diag)
    plot_matrix(y_diag, osp.join(dir, 'y_diag_cut.png'), vmax = 2, cmap = 'blue-red')

def main2():
    dir = '/home/erschultz/dataset_test/samples'
    sample = 'sample91'
    sample_dir = osp.join(dir, sample)
    sc_contacts = load_sc_contacts(sample_dir, gaussian = True, jobs = 8, sparsify = True,
                                    triu = True, correct_diag = True, down_sampling = 100)
    # sc_contacts_full = load_sc_contacts(sample_dir, gaussian = True, jobs = 8,
    #                                 triu = False, correct_diag = True, down_sampling = 1000)
    # i=0
    # yi = triu_to_full(sc_contacts[i])
    # # print(yi)
    # # print(sc_contacts_full[i])
    # assert np.allclose(yi, sc_contacts_full[i])

    # y = crop(np.load(osp.join(dir, sample, 'y.npy')), 2000)
    # y_diag = crop(np.load(osp.join(dir, sample, 'y_diag.npy')), 2000)
    # mean_dist = genomic_distance_statistics(y, mode = 'prob')
    # print(mean_dist)
    # plt.plot(mean_dist[10:])
    # plt.show()

    # y_diag1 = diagonal_preprocessing(y, mean_dist)
    # plot_matrix(y_diag, vmax = 'max')

    # y_diag2 = diagonal_preprocessing(y[np.triu_indices(len(y))], mean_dist, triu = True)
    # y_diag3 = triu_to_full(y_diag2)
    # print(np.allclose(y_diag1, y_diag3))
    # print(np.allclose(y_diag, y_diag1))

def test_lammps_load():
    file_path = '/home/erschultz/sequences_to_contact_maps/traj'
    xyz = lammps_load(file_path)

if __name__ == '__main__':
    # main2()
    # edit_argparse()
    # debugModel('ContactGNNEnergy')
    test_lammps_load()
    # plot_fixed()
    # test_argpartition(10)
    # downsampling_test()
    # testEnergy()
