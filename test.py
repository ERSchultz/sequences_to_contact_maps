import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
import os
import os.path as osp
from neural_net_utils.base_networks import *
from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import *
from neural_net_utils.argparseSetup import *
from core_test_train import core_test_train
from scipy.io import savemat

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
            _, val_dataloader, _ = getDataLoaders(dataset, opt)
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
    parser = getBaseParser()
    opt = parser.parse_args()

    # Preprocessing
    if model_type == 'UNet':
        opt.toxx = True
        opt.toxx_mode = 'mean'
        opt.x_reshape = False

    # architecture
    opt.k = 2
    opt.crop = None
    opt.m = 1024
    opt.y_preprocessing = 'diag'
    opt.y_norm = 'instance'
    opt.loss = 'mse'

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
        opt.loss = 'BCE'
        opt.y_norm = None
        opt.message_passing='signedconv'
        opt.GNN_mode = True
        opt.output_mode = 'sequence'
        opt.hidden_sizes_list=str2list('2')
        opt.out_act = None
        opt.use_node_features = False
        opt.use_edge_weights = False
        opt.transforms=str2list('none')
        opt.pre_transforms=str2list('degree')
        opt.split_neg_pos_edges_for_feature_augmentation = True
        opt.top_k = None
        opt.sparsify_threshold = 0.176
        opt.sparsify_threshold_upper = None
        opt.relabel_11_to_00 = True
        opt.y_log_transform = True
        opt.head_architecture = 'fc'
        opt.head_hidden_sizes_list = [2]
        opt.split=[0.2,0.2,0.6]
        # opt.crop=[0,5]
        # opt.m = 5
        opt.use_bias = False
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
    opt.n_epochs = 1
    opt.lr = 1e-3
    opt.batch_size = 1
    opt.milestones = str2list('1')
    opt.gamma = 0.1

    # other
    opt.plot = True
    opt.plot_predictions = True
    opt.verbose = True
    opt.data_folder = "dataset_04_18_21"

    opt = finalizeOpt(opt, parser, True)

    opt.model_type = model_type
    model = getModel(opt)

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

def downsampling_test():
    y = torch.tensor([[10,3,1,0],[3,10,4,2],[1,4,10,6], [0,2,6,10]], dtype = torch.float32)
    print(y)
    meanDist = generateDistStats(y)
    y_diag = diagonal_preprocessing(y, meanDist)
    print(y_diag)
    print('---')
    y = torch.reshape(y,(1,1,4,4))
    y_down = F.avg_pool2d(y, 2)
    y_down = torch.reshape(y_down,(2,2))
    print(y_down)
    meanDist = generateDistStats(y_down)
    y_down_diag = diagonal_preprocessing(y_down, meanDist)
    print(y_down_diag)
    y_diag = torch.tensor(y_diag, dtype = torch.float32)
    y_diag = torch.reshape(y_diag, (1,1,4,4))
    y_diag_down = F.avg_pool2d(y_diag, 2)
    print(y_diag_down)

def main():
    # edit_argparse()
    debugModel('ContactGNN')
    # test_argpartition(10)
    # to_mat()
    # downsampling_test()


if __name__ == '__main__':
    main()
