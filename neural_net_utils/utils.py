import os
import os.path as osp
import sys
import shutil
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)

import torch
from torch.utils.data import DataLoader

import torch_geometric.utils
import torch_geometric.data

import numpy as np
import math
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import csv

import networks
from dataset_classes import *

def getModel(opt):
    if opt.model_type == 'SimpleEpiNet':
        model = networks.SimpleEpiNet(opt.m, opt.k, opt.kernel_w_list, opt.hidden_sizes_list)
    if opt.model_type == 'UNet':
        model = networks.UNet(opt.nf, opt.k, opt.channels, std_norm = opt.training_norm, out_act = opt.out_act)
    elif opt.model_type == 'DeepC':
        model = networks.DeepC(opt.m, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list, opt.training_norm, opt.act, opt.out_act)
    elif opt.model_type == 'Akita':
        model = networks.Akita(opt.m, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list_trunk,
                            opt.bottleneck,
                            opt.dilation_list_head,
                            opt.act,
                            opt.out_act,
                            opt.channels,
                            opt.training_norm,
                            opt.down_sampling)
    elif opt.model_type.startswith('GNNAutoencoder'):
        model = networks.GNNAutoencoder(opt.m, opt.node_feature_size, opt.hidden_sizes_list, opt.act, opt.head_act, opt.out_act,
                                opt.message_passing, opt.head_architecture, opt.head_hidden_sizes_list, opt.parameter_sharing)
    elif opt.model_type == 'SequenceFCAutoencoder':
        model = networks.FullyConnectedAutoencoder(opt.m * opt.k, opt.hidden_sizes_list, opt.act, opt.out_act, opt.parameter_sharing)
    elif opt.model_type == 'SequenceConvAutoencoder':
        model = networks.ConvolutionalAutoencoder(opt.m, opt.k, opt.hidden_sizes_list, opt.act, opt.out_act, conv1d = True)
    elif opt.model_type.startswith('ContactGNN'):
        model = networks.ContactGNN(opt.m, opt.node_feature_size, opt.hidden_sizes_list,
        opt.act, opt.inner_act, opt.out_act,
        opt.encoder_hidden_sizes_list, opt.update_hidden_sizes_list,
        opt.message_passing, opt.use_edge_weights,
        opt.head_architecture, opt.head_hidden_sizes_list, opt.head_act, opt.use_bias,
        opt.log_file)
    elif opt.model_type == 'seq2Energy':
        model = newtorks.seq2Energy(args.k)
    else:
        raise Exception('Invalid model type: {}'.format(opt.model_type))

    return model

## dataset functions ##
def getDataset(opt, names = False, minmax = False):
    if opt.GNN_mode:
        dataset = ContactsGraph(opt.data_folder, opt.root_name, opt.m, opt.y_preprocessing, opt.y_log_transform,
                                            opt.y_norm, opt.min_subtraction, opt.use_node_features, opt.use_edge_weights,
                                            opt.sparsify_threshold, opt.sparsify_threshold_upper, opt.top_k,
                                            opt.weighted_LDP, opt.split_neg_pos_edges, opt.degree, opt.weighted_degree,
                                            opt.split_neg_pos_edges_for_feature_augmentation,
                                            opt.transforms_processed, opt.pre_transforms_processed,
                                            opt.relabel_11_to_00, opt.output_mode, opt.crop,
                                            opt.log_file)
        opt.root = dataset.root
    elif opt.autoencoder_mode and opt.output_mode == 'sequence':
        dataset = Sequences(opt.data_folder, opt.crop, opt.x_reshape, names)
        opt.root = None
    elif opt.output_mode == 'energy':
        # opt.GNN_mode guaranteed to be False
        dataset = Sequences2Energies(opt.data_folder)
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

## data processing functions ##
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

## plotting helper functions ##
def un_normalize(y, minmax):
    ymin = minmax[0].item()
    ymax = minmax[1].item()
    return y.copy() * (ymax - ymin) + ymin

def getPercentiles(arr, prcnt_arr, plot = True):
    """Helper function to get multiple percentiles at once."""
    result = np.zeros_like(prcnt_arr).astype(np.float64)
    if plot:
        plt.hist(arr.flatten())
        plt.show()
    for i, p in enumerate(prcnt_arr):
        result[i] = np.percentile(arr, p)
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

    overall_corr, _ = stat(y[triu_ind], yhat[triu_ind])

    corr_arr = np.zeros(n-2)
    corr_arr[0] = np.NaN
    for d in range(1, n-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = stat(y_diag, yhat_diag)
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
            assert opt.autoencoder_mode
            y = data.contact_map
            yhat = model(data)
            yhat = torch.reshape(yhat, (opt.m, opt.m))
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
        y = y.cpu().numpy().reshape((opt.m, opt.m))
        y = un_normalize(y, minmax)
        if opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt':
            ytrue = y
            yhat = np.argmax(yhat, axis = 1)
        if opt.y_preprocessing == 'diag':
            ytrue = np.load(osp.join(path, 'y_prcnt.npy'))
            yhat = un_normalize(yhat, minmax)
            yhat = percentile_preprocessing(yhat, prcntDist)
        yhat = yhat.reshape((opt.m,opt.m))
        acc = np.sum(yhat == ytrue) / yhat.size
        acc_arr[i] = acc

        for c in range(opt.classes):
            denom = np.sum(ytrue == c)
            freq_c_arr[i, c] = denom / ytrue.size
            num = np.sum(np.logical_and((ytrue == c), (yhat == ytrue)))
            acc = num / denom
            acc_c_arr[i, c] = acc

    acc_result = 'Accuracy: {}% +- {}'.format(round(np.mean(acc_arr) * 100, 3) , round(np.std(acc_arr) * 100, 3) )
    print(acc_result, file = opt.log_file)
    print('Loss: {} +- {}'.format(np.round(np.mean(loss_arr), 3), np.round( np.std(loss_arr), 3)), file = opt.log_file)
    return acc_c_arr, freq_c_arr, acc_result

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
            assert opt.autoencoder_mode
            y = data.contact_map
            yhat = model(data)
            path = data.path[0]
            minmax = data.minmax
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
        yhat = yhat.reshape((opt.m, opt.m))

        # y
        pca.fit(y)
        comp1_y = pca.components_[0]
        sign1_y = np.sign(comp1_y)

        # yhat
        pca.fit(yhat)
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
            plt.title('PC 1\nPearson R: corr')
            plt.savefig(osp.join(subpath, 'pc1.png'))
            plt.close()

    results = 'PCA Results:\n' +\
            'Accuracy: {} +- {}\n'.format(np.round(np.mean(acc_arr), 3), np.round(np.std(acc_arr), 3)) +\
            'Spearman R: {} +- {}\n'.format(np.round(np.mean(rho_arr), 3), np.round(np.std(rho_arr), 3))+\
            'Pearson R: {} +- {}\n'.format(np.round(np.mean(p_arr), 3), np.round(np.std(p_arr), 3))
    print(results, file = opt.log_file)
    with open(osp.join(imagePath, 'PCA_results.txt'), 'w') as f:
        f.write(results)

def roundUpBy10(val):
    """Rounds value up to the nearst multiple of 10."""
    assert val > 0, "val too small"
    assert val < 10**10, "val too big"
    mult = 1
    while val > mult:
        mult *= 10
    return mult

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

        if self.chi is not None:
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
    aggregate_peaks('chip_seq_data/ENCFF101KOJ.bed')
    # copy_data_to_scratch(opt)
    # plotPerClassAccuracy(None, None, 5)

if __name__ == '__main__':
    main()
