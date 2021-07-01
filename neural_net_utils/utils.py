import os
import os.path as osp
import sys
import shutil
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.utils
import torch_geometric.data
import torch_geometric.transforms

import numpy as np
import math
import argparse
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import csv
from networks import *
from dataset_classes import *

def getModel(opt):
    if opt.model_type == 'SimpleEpiNet':
        model = SimpleEpiNet(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list)
    if opt.model_type == 'UNet':
        model = UNet(opt.nf, opt.k, opt.channels, std_norm = opt.training_norm, out_act = opt.out_act)
    elif opt.model_type == 'DeepC':
        model = DeepC(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list, opt.training_norm, opt.out_act)
    elif opt.model_type == 'Akita':
        model = Akita(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list_trunk,
                            opt.bottleneck,
                            opt.dilation_list_head,
                            opt.out_act,
                            opt.channels,
                            opt.training_norm,
                            opt.down_sampling)
    elif opt.model_type == 'GNNAutoencoder':
        model = GNNAutoencoder(opt.n, opt.node_feature_size, opt.hidden_sizes_list, opt.out_act,
                                opt.message_passing, opt.head_architecture, opt.MLP_hidden_sizes_list)
    elif opt.model_type == 'SequenceFCAutoencoder':
        model = FullyConnectedAutoencoder(opt.n * opt.k, opt.hidden_sizes_list)
    elif opt.model_type == 'ContactGNN':
        model = ContactGNN(opt.n, opt.node_feature_size, opt.hidden_sizes_list, opt.out_act, opt.message_passing)
    else:
        raise Exception('Invalid model type: {}'.format(opt.model_type))

    return model

# dataset functions
def getDataset(opt, names = False, minmax = False):
    if opt.GNN_mode:
        dataset = ContactsGraph(opt.data_folder, opt.root_name, opt.n, opt.y_preprocessing,
                                            opt.y_norm, opt.min_subtraction, opt.use_node_features,
                                            transform = opt.transforms_processed)
        opt.root = dataset.root
    elif opt.autoencoder_mode and opt.output_mode == 'sequence':
        dataset = Sequences(opt.data_folder, opt.crop, names)
        opt.root = None
    else:
        dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.toxx_mode, opt.y_preprocessing,
                                            opt.y_norm, opt.x_reshape, opt.ydtype,
                                            opt.y_reshape, opt.crop, opt.min_subtraction, names, minmax)
        opt.root = None
    return dataset

def getDataLoaders(dataset, opt):
    train_dataset, val_dataset, test_dataset = splitDataset(dataset, opt)

    if opt.GNN_mode:
        dataloader_fn = torch_geometric.data.DataLoader
    else:
        dataloader_fn = DataLoader

    train_dataloader = dataloader_fn(train_dataset, batch_size = opt.batch_size,
                                    shuffle = opt.shuffle, num_workers = opt.num_workers)
    if len(val_dataset) > 0:
        val_dataloader = dataloader_fn(val_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        val_dataloader = None
    if len(val_dataset) > 0:
        test_dataloader = dataloader_fn(test_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader

def splitDataset(dataset, opt):
    """Splits input dataset into proportions specified by split."""
    opt.N = len(dataset)
    assert sum(opt.split) - 1 < 1e-5, "split doesn't sum to 1: {}".format(opt.split)
    opt.trainN = math.floor(opt.N * opt.split[0])
    opt.valN = math.floor(opt.N * opt.split[1])
    opt.testN = opt.N - opt.trainN - opt.valN
    return torch.utils.data.random_split(dataset, [opt.trainN, opt.valN, opt.testN], torch.Generator().manual_seed(opt.seed))

# data processing functions
def x2xx(x, mode = 'mean'):
    # TODO better explanation here
    """
    Function for converting x to an image xx.

    For add:
        xx[k, i, j] = x[i, :] + x[j, :] (element-wise)
    For mean:
        xx[k, i, j] = (x[i, :] + x[j, :]) / 2 (element-wise)
    For append:
        xx[2k, i, j] = [x[i, :] ... x[j, :]]


    Inputs:
        x: (n x k) array
        mode: method for combining xi and xj

    Outputs:
        xx: (2k x n x n) or (k x n x n) array
    """
    n, k = x.shape
    if mode == 'append':
        xx = np.zeros((k*2, n, n))
    else:
        xx = np.zeros((k, n, n))
    for i in range(n):
        for j in range(i+1):
            if mode == 'append':
                xx[:, i, j] = np.append(x[i], x[j])
                xx[:, j, i] = np.append(x[j], x[i])
            elif mode =='add':
                xx[:, i, j] = np.add(x[i], x[j])
                xx[:, j, i] = np.add(x[j], x[i])
            elif mode == 'mean':
                xx[:, i, j] = np.add(x[i], x[j]) / 2
                xx[:, j, i] = np.add(x[j], x[i]) / 2
    return xx

def diagonal_preprocessing(y, meanDist):
    """
    Removes diagonal effect from contact map y.

    Inputs:
        y: contact map numpy array
        mean: mean contact frequency where mean[dist] is the mean at a given distance

    Outputs:
        result: new contact map
    """
    result = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(i + 1):
            distance = i - j
            exp_d = meanDist[distance]
            if exp_d == 0:
                # this is unlikely to happen
                pass
            else:
                result[i,j] = y[i,j] / exp_d
                result[j,i] = result[i,j]

    return result

def percentile_preprocessing(y, percentiles):
    """
    Performs percentile preprocessing on contact map y.

    The maximum value in y must at most percentiles[-1].

    Inputs:
        y: contact map numpy array (2 dimensional)
        percentiles: list of percentiles (frequencies)

    Outputs:
        result: new contact map
    """

    if len(y.shape) == 3:
        N, H, W = y.shape
        assert N == 1
        y = y.reshape(H,W)
    elif len(y.shape) == 4:
        N, C, H, W = y.shape
        assert N == 1 and C == 1
        y = y.reshape(H,W)

    result = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(i + 1):
            val = y[i,j]
            p = 0
            while p < len(percentiles)-1 and percentiles[p] < val:
                p += 1
            result[i,j] = p
            result[j,i] = p

    return result

# plotting helper functions
def un_normalize(y, minmax):
    ymin = minmax[0].item()
    ymax = minmax[1].item()
    return y.copy() * (ymax - ymin) + ymin

def getFrequencies(dataFolder, diag, n, k, chi):
    # calculates number of times each interaction frequency
    # was observed
    converter = InteractionConverter(k)
    samples = make_dataset(dataFolder)
    freq_arr = np.zeros((int(n * (n+1) / 2 * len(samples)), 4)) # freq, sample, type, psi_ij
    ind = 0
    for sample in samples:

        sampleid = int(osp.split(sample)[-1][6:])

        x = np.load(osp.join(sample, 'x.npy'))
        if diag:
            y = np.load(osp.join(sample, 'y_diag.npy'))
        else:
            y = np.load(osp.join(sample, 'y.npy'))
        for i in range(n):
            xi = x[i]
            for j in range(i+1):
                xj = x[j]
                comb = frozenset({tuple(xi), tuple(xj)})
                comb_type = converter.comb2Type(comb)
                psi_ij = xi @ chi @ xj
                freq_arr[ind] = [y[i,j], sampleid, comb_type, psi_ij]
                ind += 1

    return freq_arr

def getPercentiles(arr, prcnt_arr):
    """Helper function to get multiple percentiles at once."""
    result = np.zeros_like(prcnt_arr).astype(np.float64)
    plt.hist(arr.flatten())
    plt.show()
    for i, p in enumerate(prcnt_arr):
        result[i] = np.percentile(arr, p)
    return result

def generateDistStats(y, mode = 'freq', stat = 'mean'):
    '''
    Calculates statistics of contact frequency/probability as a function of distance

    Inputs:
        mode: freq for frequencies, prob for probabilities
        stat: mean to calculate mean, var for variance

    Outputs:
        result: numpy array where result[d] is the contact frequency/probability stat at distance d
    '''
    if mode == 'prob':
        y = y.copy() / np.max(y)

    if stat == 'mean':
        npStat = np.mean
    elif stat == 'var':
        npStat = np.var
    n = len(y)
    distances = range(0, n, 1)
    result = np.zeros_like(distances).astype(float)
    for d in distances:
        result[d] = npStat(np.diagonal(y, offset = d))

    return result

def calculateDistanceStratifiedCorrelation(y, yhat, mode = 'pearson'):
    """
    Helper function to calculate correlation stratified by distance.

    Inputs:
        y: target
        yhat: prediction
        mode: pearson or spearman (str)

    Outpus:
        overall_corr: overall correlation
        corr_arr: array of distance stratified correlations
    """
    if mode.lower() == 'pearson':
        stat = pearsonr
    elif mode.lower() == 'spearman':
        stat = spearmanr

    assert len(y.shape) == 2
    n, n = y.shape
    triu_ind = np.triu_indices(n)

    overall_corr, pval = pearsonr(y[triu_ind], yhat[triu_ind])

    corr_arr = np.zeros(n-1)
    for d in range(n-1):
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, pval = stat(y_diag, yhat_diag)
        corr_arr[d] = corr

    return overall_corr, corr_arr

def calculatePerClassAccuracy(val_dataloader, model, opt):
    print('Class Accuracy Results:', file = opt.log_file)
    assert opt.y_preprocessing in {'diag', 'prcnt'}, "invalid preprocessing: {}".format(opt.y_preprocessing)
    if opt.y_preprocessing != 'prcnt':
        prcntDist_path = osp.join(opt.data_folder, 'prcntDist.npy')
        prcntDist = np.load(prcntDist_path)
        print('prcntDist', prcntDist, file = opt.log_file)

    loss_arr = np.zeros(opt.valN)
    acc_arr = np.zeros(opt.valN)
    acc_c_arr = np.zeros((opt.valN, opt.classes))
    freq_c_arr = np.zeros((opt.valN, opt.classes))

    for i, data in enumerate(val_dataloader):
        if opt.GNN_mode:
            data = data.to(opt.device)
            if opt.autoencoder_mode:
                y = torch.reshape(torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr), (opt.n, opt.n))
            else:
                y = data.y
            yhat = model(data)
            path = data.path[0]
            minmax = data.minmax
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        loss_arr[i] = loss
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        y = un_normalize(y, minmax)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt':
            ytrue = y
            yhat = np.argmax(yhat, axis = 1)
        if opt.y_preprocessing == 'diag':
            ytrue = np.load(osp.join(path, 'y_prcnt.npy'))
            yhat = un_normalize(yhat, minmax)
            yhat = percentile_preprocessing(yhat, prcntDist)
        yhat = yhat.reshape((opt.n,opt.n))
        acc = np.sum(yhat == ytrue) / yhat.size
        acc_arr[i] = acc

        for c in range(opt.classes):
            denom = np.sum(ytrue == c)
            freq_c_arr[i, c] = denom / ytrue.size
            num = np.sum(np.logical_and((ytrue == c), (yhat == ytrue)))
            acc = num / denom
            acc_c_arr[i, c] = acc

    acc_result = 'Accuracy: {}% +- {}'.format(round(np.mean(acc_arr), 3) * 100, round(np.std(acc_arr), 3) * 100)
    print(acc_result, file = opt.log_file)
    print('Loss: {} +- {}'.format(np.round(np.mean(loss_arr), 3), np.round( np.std(loss_arr), 3)), file = opt.log_file)
    return acc_c_arr, freq_c_arr, acc_result


# other functions
def comparePCA(val_dataloader, imagePath, model, opt, count = 5):
    """Computes statistics of 1st PC of contact map"""
    acc_arr = np.zeros(opt.valN)
    rho_arr = np.zeros(opt.valN)
    p_arr = np.zeros(opt.valN)
    pca = PCA()
    model.eval()
    for i, data in enumerate(val_dataloader):
        if opt.GNN_mode:
            data = data.to(opt.device)
            if opt.autoencoder_mode:
                y = torch.reshape(torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr), (opt.n, opt.n))
            else:
                y = data.y
            yhat = model(data)
            path = data.path[0]
            minmax = data.minmax
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        y = un_normalize(y, minmax)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt' and opt.loss == 'cross_entropy':
            yhat = np.argmax(yhat, axis = 1)
        else:
            yhat = un_normalize(yhat, minmax)
        yhat = yhat.reshape((opt.n, opt.n))

        # y
        result_y = pca.fit(y)
        comp1_y = pca.components_[0]
        sign1_y = np.sign(comp1_y)

        # yhat
        result_yhat = pca.fit(yhat)
        comp1_yhat = pca.components_[0]
        sign1_yhat = np.sign(comp1_yhat)

        # results
        acc = np.sum((sign1_yhat == sign1_y)) / sign1_y.size
        acc_arr[i] = max(acc, 1 - acc)

        corr, pval = spearmanr(comp1_yhat, comp1_y)
        rho_arr[i] = abs(corr)

        corr, pval = pearsonr(comp1_yhat, comp1_y)
        p_arr[i] = abs(corr)

        # for plotting
        if i < count:
            sample = osp.split(path)[-1]
            subpath = osp.join(opt.ofile_folder, sample)
            if not osp.exists(subpath):
                os.mkdir(subpath, mode = 0o755)

            plt.plot(comp1_yhat, label = 'yhat')
            plt.plot(comp1_y, label = 'y')
            plt.legend()
            plt.title('PC 1')
            plt.savefig(osp.join(subpath, 'pc1.png'))
            plt.close()

    results = 'PCA Results:\n' +\
            'Accuracy: {} +- {}\n'.format(np.round(np.mean(acc_arr), 3), np.round(np.std(acc_arr), 3)) +\
            'Spearman R: {} +- {}\n'.format(np.round(np.mean(rho_arr), 3), np.round(np.std(rho_arr), 3))+\
            'Pearson R: {} +- {}\n'.format(np.round(np.mean(p_arr), 3), np.round(np.std(p_arr), 3))
    print(results, file = opt.log_file)
    with open(osp.join(imagePath, 'PCA_results.txt'), 'w') as f:
        f.write(results)

def getBaseParser():
    '''Helper function that returns base parser'''
    parser = argparse.ArgumentParser(description='Base parser', fromfile_prefix_chars='@')
    parser.add_argument('--GNN_mode', type=str2bool, default=False, help='True to use GNNs (uses pytorch_geometric in core_test_train)')
    parser.add_argument('--autoencoder_mode', type=str2bool, default=False, help='True to use input as target output (i.e. autoencoder)')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--output_mode', type=str, default='contact', help='data structure of output {"contact", "sequence"}')

    # pre-processing args
    parser.add_argument('--data_folder', type=str, default='dataset_04_18_21', help='Location of data')
    parser.add_argument('--root_name', type=str, help='name of file to save graph data (leave as None to create root automatically) (root is the directory path - defined later)')
    parser.add_argument('--delete_root', type=str2bool, default=True, help='True to delete root directory after runtime')
    parser.add_argument('--toxx', type=str2bool, default=False, help='True if x should be converted to 2D image')
    parser.add_argument('--toxx_mode', type=str, default='mean', help='mode for toxx (default mean)')
    parser.add_argument('--y_preprocessing', type=str2None, default='diag', help='type of pre-processing for y')
    parser.add_argument('--y_norm', type=str2None, default='batch', help='type of [0,1] normalization for y')
    parser.add_argument('--min_subtraction', type=str2bool, default=True, help='if min subtraction should be used for y_norm')
    parser.add_argument('--x_reshape', type=str2bool, default=True, help='True if x should be considered a 1D image')
    parser.add_argument('--ydtype', type=str2dtype, default='float32', help='torch data type for y')
    parser.add_argument('--y_reshape', type=str2bool, default=True, help='True if y should be considered a 2D image')
    parser.add_argument('--crop', type=str2list, help='size of crop to apply to image - format: <leftcrop-rightcrop>')
    parser.add_argument('--classes', type=int, default=10, help='number of classes in percentile normalization')
    parser.add_argument('--use_scratch', type=str2bool, default=False, help='True to move data to scratch')
    parser.add_argument('--use_node_features', type=str2bool, default=False, help='True to use node features for GNN models')

    # dataloader args
    parser.add_argument('--split', type=str2list, default=[0.8, 0.1, 0.1], help='Train, val, test split for dataset')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Whether or not to shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5, help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2, help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning eate. Default=0.001')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--milestones', type=str2list, default=[2], help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr decay')
    parser.add_argument('--loss', type=str, default='mse', help='Type of loss to use: options: {"mse", "cross_entropy"}')

    # model args
    parser.add_argument('--model_type', type=str, default='test', help='Type of model')
    parser.add_argument('--id', type=int, help='id of model')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='True if using a pretrained model')
    parser.add_argument('--resume_training', type=str2bool, default=False, help='True if resuming traning of a partially trained model')
    parser.add_argument('--ifile_folder', type=str, help='Location of input file for pretrained model')
    parser.add_argument('--ifile', type=str, help='Name of input file for pretrained model')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')
    parser.add_argument('--out_act', type=str2None, help='activation of final layer')
    parser.add_argument('--training_norm', type=str2None, help='norm during training (batch, instance, or None)')

    # SimpleEpiNet args
    parser.add_argument('--kernel_w_list', type=str2list, help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=str2list, help='List of hidden sizes for convolutional layers')

    # UNet args
    parser.add_argument('--nf', type=int, help='Number of filters')

    # DeepC args
    parser.add_argument('--dilation_list', type=str2list, help='List of dilations for dilated convolutional layers')

    # Akita args
    parser.add_argument('--dilation_list_trunk', type=str2list, help='List of dilations for dilated convolutional layers of trunk')
    parser.add_argument('--bottleneck', type=int, help='Number of filters in bottleneck (must be <= hidden_size_dilation_trunk)')
    parser.add_argument('--dilation_list_head', type=str2list, help='List of dilations for dilated convolutional layers of head')
    parser.add_argument('--down_sampling', type=str2None, help='type of down sampling to use')

    # GNNAutoencoder args
    parser.add_argument('--message_passing', type=str, default='GCN', help='type of message passing algorithm')
    parser.add_argument('--head_architecture', type=str, default= 'xxT', help='type of head architecture')
    parser.add_argument('--MLP_hidden_sizes_list', type=str2list, help='List of hidden sizes for convolutional layers')
    parser.add_argument('--transforms', type=str2list, help='list of transforms to use for GNN')


    # post-processing args
    parser.add_argument('--plot', type=str2bool, default=True, help='True to plot result figures')
    parser.add_argument('--plot_predictions', type=str2bool, default=True, help='True to plot predictions')

    return parser

def finalizeOpt(opt, parser, local = False):
    print(opt)
    # local is a flag to overide some commands when working locally
    # set up output folders/files
    model_type_folder = osp.join('results', opt.model_type)
    if opt.id is None:
        if not osp.exists(model_type_folder):
            os.mkdir(model_type_folder, mode = 0o755)
            opt.id = 1
        else:
            max_id = 0
            for filename in os.listdir(model_type_folder):
                if filename.isnumeric():
                    id = int(filename)
                    if id > max_id:
                        max_id = id
            opt.id = max_id + 1
    else:
        txt_file = osp.join(model_type_folder, str(opt.id), 'argparse.txt')
        assert osp.exists(txt_file), "{} does not exist".format(txt_file)
        id_copy = opt.id
        opt = parser.parse_args(sys.argv.insert(1, '@{}'.format(txt_file))) # parse again
        # by inserting at position 1, the original arguments will override the txt file
        print(sys.argv)
        opt.id = id_copy



    opt.ofile_folder = osp.join(model_type_folder, str(opt.id))
    if not osp.exists(opt.ofile_folder):
        os.mkdir(opt.ofile_folder, mode = 0o755)
    log_file_path = osp.join(opt.ofile_folder, 'out.log')
    opt.log_file = open(log_file_path, 'a')

    # configure other model params
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
        opt.channels = 1
    elif opt.loss == 'cross_entropy':
        assert opt.out_act is None, "Cannot use output activation with cross entropy"
        assert not opt.GNN_mode, 'cross_entropy not validated for GNN'
        assert opt.y_preprocessing == 'prcnt', 'must use percentile preprocessing with cross entropy'
        assert opt.y_norm is None, 'Cannot normalize with cross entropy'
        opt.channels = opt.classes
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
    elif opt.loss == 'BCE':
        assert opt.out_act is None, "Cannot use output activation with BCE"
        assert opt.y_norm is not None, 'must use some sort of y_norm'
        opt.criterion = F.binary_cross_entropy_with_logits
    else:
        raise Exception('Invalid loss: {}'.format(repr(opt.loss)))

    # check mode
    if opt.model_type == 'GNNAutoencoder':
        assert opt.GNN_mode, 'mode should be GNN for GNNAutoencoder'

    # configure GNN transforms
    opt.node_feature_size = 0
    if opt.use_node_features:
        opt.node_feature_size += opt.k
    if opt.transforms is not None:
        transforms_processed = []
        for t_str in opt.transforms:
            if t_str.lower() == 'constant':
                transforms_processed.append(torch_geometric.transforms.Constant())
                opt.node_feature_size += 1
            elif t_str.lower() == 'onehotdegree':
                transforms_processed.append(torch_geometric.transforms.OneHotDegree(opt.n))
                opt.node_feature_size += opt.n
            else:
                raise Exception("Invalid transform {}".format(t_str))
        opt.transforms_processed = torch_geometric.transforms.Compose(transforms_processed)
    else:
        opt.transforms_processed = None

    # move data to scratch
    if opt.use_scratch and not local:
        copy_data_to_scratch(opt)

    # configure cuda
    if opt.gpus > 1:
        opt.cuda = True
        opt.use_parallel = True
        opt.gpu_ids = []
        for ii in range(6):
            try:
                torch.cuda.get_device_properties(ii)
                print(str(ii), file = opt.log_file)
                opt.gpu_ids.append(ii)
            except AssertionError:
                print('Not ' + str(ii) + "!", file = opt.log_file)
    elif opt.gpus == 1:
        opt.cuda = True
        opt.use_parallel = False
    else:
        opt.cuda = False
        opt.use_parallel = False

    if opt.cuda and not torch.cuda.is_available():
        if not local:
            print('Warning: falling back to cpu', file = opt.log_file)
        opt.cuda = False
        opt.use_parallel = False

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # Set random seeds
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    return opt

def copy_data_to_scratch(opt):
    scratch_path = osp.join('/../../../scratch/midway2/erschultz', osp.split(opt.data_folder)[-1])
    if not osp.exists(scratch_path):
        os.mkdir(scratch_path, mode = 0o700)

    for file in os.listdir(opt.data_folder):
        file_dir = osp.join(opt.data_folder, file)
        scratch_file_dir = osp.join(scratch_path, file)
        if file.endswith('npy') and not osp.exists(scratch_file_dir):
            shutil.copyfile(file_dir, scratch_file_dir)

    if not osp.exists(osp.join(scratch_path, 'samples')):
        os.mkdir(osp.join(scratch_path, 'samples'), mode = 0o700)

    for sample in os.listdir(osp.join(opt.data_folder, 'samples')):
        sample_dir = osp.join(opt.data_folder, 'samples', sample)
        scratch_sample_dir = osp.join(scratch_path, 'samples', sample)
        if not osp.exists(scratch_sample_dir):
            os.mkdir(scratch_sample_dir, mode = 0o700)
        for file in os.listdir(sample_dir):
            file_dir = osp.join(sample_dir, file)
            scratch_file_dir = osp.join(scratch_sample_dir, file)
            if file.endswith('npy') and not osp.exists(scratch_file_dir):
                shutil.copyfile(file_dir, scratch_file_dir)

    opt.data_folder = scratch_path


def argparseSetup():
    """Helper function set up parser."""
    parser = getBaseParser()
    opt = parser.parse_args()
    return finalizeOpt(opt, parser)


def save_args(opt):
    with open(osp.join(opt.ofile_folder, 'argparse.txt'), 'w') as f:
        for arg in sys.argv[1:]: # skip the program file
            f.write(arg + '\n')

def opt2list(opt):
    opt_list = [opt.model_type, opt.id, opt.data_folder, opt.toxx, opt.toxx_mode, opt.y_preprocessing,
        opt.y_norm, opt.x_reshape, opt.ydtype, opt.y_reshape, opt.crop, opt.classes, opt.split,
        opt.shuffle, opt.batch_size, opt.num_workers, opt.start_epoch, opt.n_epochs, opt.lr,
        opt.gpus, opt.milestones, opt.gamma, opt.loss, opt.pretrained, opt.resume_training,
        opt.ifile_folder, opt.ifile, opt.k, opt.n, opt.seed, opt.out_act, opt.training_norm,
        opt.plot, opt.plot_predictions]
    if opt.GNN_mode:
        opt_list.extend([opt.use_node_features, opt.transforms])
    if opt.model_type == 'simpleEpiNet':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list])
    elif opt.model_type == 'UNet':
        opt_list.append(opt.nf)
    elif opt.model_type == 'Akita':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list_trunk, opt.bottleneck, opt.dilation_list_head, opt.down_sampling])
    elif opt.model_type == 'DeepC':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list])
    elif opt.model_type == 'test':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list_trunk, opt.bottleneck, opt.dilation_list_head, opt.nf])
    elif opt.model_type == 'GNNAutoencoder':
        opt_list.extend([opt.hidden_sizes_list, opt.message_passing, opt.head_architecture])
    else:
        raise Exception("Unknown model type: {}".format(opt.model_type))

    return opt_list

def save_opt(opt, ofile):
    if not osp.exists(ofile):
        with open(ofile, 'w', newline = '') as f:
            wr = csv.writer(f)
            opt_list = get_opt_header(opt.model_type, opt.mode)
            wr.writerow(opt_list)
    with open(ofile, 'a') as f:
        wr = csv.writer(f)
        opt_list = opt2list(opt)
        wr.writerow(opt_list)

def get_opt_header(model_type, mode = None):
    opt_list = ['model_type', 'id',  'data_folder','toxx', 'toxx_mode', 'y_preprocessing',
        'y_norm', 'x_reshape', 'ydtype', 'y_reshape', 'crop', 'classes', 'split', 'shuffle',
        'batch_size', 'num_workers', 'start_epoch', 'n_epochs', 'lr', 'gpus', 'milestones',
        'gamma', 'loss', 'pretrained', 'resume_training', 'ifile_folder', 'ifile', 'k', 'n',
        'seed', 'out_act', 'training_norm', 'plot', 'plot_predictions']
    if mode == 'GNN':
        opt_list.extend(['use_node_features','transforms'])
    if model_type == 'simpleEpiNet':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list'])
    elif model_type == 'UNet':
        opt_list.append('nf')
    elif model_type == 'Akita':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list_trunk', 'bottleneck', 'dilation_list_head', 'down_sampling'])
    elif model_type == 'DeepC':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list'])
    elif model_type == 'test':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list_trunk', 'bottleneck', 'dilation_list_head', 'nf'])
    elif model_type == 'GNNAutoencoder':
        opt_list.extend(['hidden_sizes_list', 'message_passing', 'head_architecture'])
    else:
        raise Exception("Unknown model type: {}".format(model_type))

    return opt_list


def roundUpBy10(val):
    """Rounds value up to the nearst multiple of 10."""
    assert val > 0, "val too small"
    assert val < 10**10, "val too big"
    mult = 1
    while val > mult:
        mult *= 10
    return mult

def str2None(v):
    """
    Helper function for argparser, converts str to None if str == 'none'

    Inputs:
        v: string
    """
    if v is None:
        return v
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        else:
            return v
    else:
        raise argparse.ArgumentTypeError('String value expected.')

def str2bool(v):
    """
    Helper function for argparser, converts str to boolean for various string inputs.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Inputs:
        v: string
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v, sep = '-'):
    """
    Helper function for argparser, converts str to list by splitting on sep.

    Exmaple for sep = '-': "i-j-k" -> [i,j,k]

    Inputs:
        v: string
        sep: separartor
    """
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        else:
            result = [i for i in v.split(sep)]
            for i, val in enumerate(result):
                if val.isnumeric():
                    result[i] = int(val)
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def str2dtype(v):
    """
    Helper function for argparser, converts str to torch dtype.

    Inputs:
        v: string
    """
    if isinstance(v, str):
        if v == 'float32':
            return torch.float32
        elif v == 'float64':
            return torch.float64
        elif v == 'int32':
            return torch.int32
        elif v == 'int64':
            return torch.int64
        else:
            raise Exception('Unkown str: {}'.format(v))
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def list2str(v, sep = '-'):
    """
    Helper function to convert list to string.

    Inputs:
        v: list
    """
    if isinstance(v, list):
        return sep.join([str(i) for i in v])
    else:
        raise Exception('list value expected.')

def float2str(v):
    """
    Helper function to convert float to str in si notation.

    Inputs:
        v: float
    """
    # TODO make this more robust
    if isinstance(v, float):
        vstr = "{:.1e}".format(v)
        if vstr[2] == '0':
            # converts 1.0e-04 to 1e-04
            vstr = vstr[0:1] + vstr[3:]
        if vstr[-2] == '0':
            # converts 1e-04 to 1e-4
            vstr = vstr[0:-2] + vstr[-1]
    else:
        raise Exception('float value expected.')
    return vstr

class InteractionConverter():
    """Class that allows conversion between epigenetic mark bit string pairs and integer type id"""
    def __init__(self, k, chi = None):
        self.k = k
        self.chi = chi
        self.allStrings = self.generateAllBinaryStrings()
        self.comb2TypeDict = {}
        self.type2CombDict = {}

        curr_type = 0
        n = len(self.allStrings)
        for i in range(n):
            xi = self.allStrings[i]
            for j in range(n):
                xj = self.allStrings[j]
                comb = frozenset({tuple(xi), tuple(xj)})
                if comb not in self.comb2TypeDict.keys():
                    self.comb2TypeDict[comb] = curr_type
                    self.type2CombDict[curr_type] = comb
                    curr_type += 1

        self.types = np.arange(0, curr_type, 1)

        if self.chi is not None and self.x is not None:
            self.setPsi()

    def setChi(self, chi):
        self.chi = chi

    def setPsi(self):
        assert self.chi is not None, "set chi first"
        self.Psi = self.allStrings @ self.chi @ self.allStrings.T

    def getPsi_ij(self, xi, xj):
        assert self.chi is not None, "set chi first"
        return xi @ self.chi @ xj

    def comb2Type(self, comb):
        # input comb must be a frozenset
        return self.comb2TypeDict[comb]

    def type2Comb(self, type):
        # input type must be an integer
        return self.type2CombDict[type]

    def comb2str(self, comb):
        combList = sorted(list(comb))
        if len(combList) == 2:
            return '{} - {}'.format(combList[0], combList[1])
        else:
            return '{} - {}'.format(combList[0], combList[0])

    def fillArrayWithAllBinaryStrings(self, n, arr, temp_arr, i, row = 0):
        # https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/
        if i == n:
            arr.append(temp_arr.copy())
            row += 1
            return row

        # First assign "1" at ith position
        # and try for all other permutations
        # for remaining positions
        temp_arr[i] = 1
        self.fillArrayWithAllBinaryStrings(n, arr, temp_arr, i + 1, row)

        # And then assign "0" at ith position
        # and try for all other permutations
        # for remaining positions
        temp_arr[i] = 0
        self.fillArrayWithAllBinaryStrings(n, arr, temp_arr, i + 1, row)

    def generateAllBinaryStrings(self):
        arr = []
        temp_arr = [None]*self.k
        self.fillArrayWithAllBinaryStrings(self.k, arr, temp_arr, 0)
        np_arr = np.array(arr).astype(np.int8)
        return np_arr

def main():
    opt = argparseSetup()
    print(opt)
    copy_data_to_scratch(opt)
    # plotPerClassAccuracy(None, None, 5)

if __name__ == '__main__':
    main()
