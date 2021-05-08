import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from numba import jit, njit
import numpy as np
import os
import math
import argparse
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# dataset functions
def make_dataset(dir, minSample = 0):
    data_file_arr = []
    samples_dir = os.path.join(dir, 'samples')
    for file in os.listdir(samples_dir):
        if not file.startswith('sample'):
            print("Skipping {}".format(file))
        else:
            sample_id = int(file[6:])
            if sample_id < minSample:
                print("Skipping {}".format(file))
            else:
                data_file = os.path.join(samples_dir, file)
                data_file_arr.append(data_file)
    return data_file_arr

def getDataLoaders(dataset, opt):
    train_dataset, val_dataset, test_dataset = splitDataset(dataset, opt)
    print()
    train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size,
                                    shuffle = opt.shuffle, num_workers = opt.num_workers)
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        val_dataloader = None
    if len(val_dataset) > 0:
        test_dataloader = DataLoader(test_dataset, batch_size = opt.batch_size,
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
@njit
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
        max: maximum value of

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

    if len(y.shape) > 2:
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
def getFrequencies(dataFolder, diag, n, k, chi):
    # calculates number of times each interaction frequency
    # was observed
    converter = InteractionConverter(k)
    samples = make_dataset(dataFolder)
    freq_arr = np.zeros((int(n * (n+1) / 2 * len(samples)), 4)) # freq, sample, type, psi_ij
    ind = 0
    for sample in samples:

        sampleid = int(os.path.split(sample)[-1][6:])

        x = np.load(os.path.join(sample, 'x.npy'))
        if diag:
            y = np.load(os.path.join(sample, 'y_diag.npy'))
        else:
            y = np.load(os.path.join(sample, 'y.npy'))
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
    assert opt.y_preprocessing == 'diag' or opt.y_preprocessing == 'prcnt', "invalid preprocessing: {}".format(opt.y_preprocessing)
    if opt.y_preprocessing != 'prcnt':
        prcntDist_path = os.path.join(opt.data_folder, 'prcntDist.npy')
        prcntDist = np.load(prcntDist_path)

    loss_arr = np.zeros(opt.valN)
    acc_arr = np.zeros(opt.valN)
    acc_c_arr = np.zeros((opt.valN, opt.classes))
    freq_c_arr = np.zeros((opt.valN, opt.classes))

    for i, (x, y, path, max) in enumerate(val_dataloader):
        path = path[0]
        ymax = max.item()
        yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        loss_arr[i] = loss
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt':
            ytrue = y
            yhat = np.argmax(yhat, axis = 1)
        if opt.y_preprocessing == 'diag':
            ytrue = np.load(os.path.join(path, 'y_prcnt.npy'))
            if opt.y_norm == 'instance':
                yhat = yhat * ymax
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

    print('Accuracy: {} +- {}'.format(np.mean(acc_arr), np.std(acc_arr)))
    print('Loss: {} +- {}'.format(np.mean(loss_arr), np.std(loss_arr)))
    print('acc arr', acc_c_arr)
    print('freq arr', freq_c_arr)
    return acc_c_arr, freq_c_arr

# other functions
def comparePCA(val_dataloader, model, opt):
    """Computes statistics of 1st PC of contact map"""
    acc_arr = np.zeros(opt.valN)
    rho_arr = np.zeros(opt.valN)
    p_arr = np.zeros(opt.valN)
    pca = PCA()
    model.eval()
    for i, (x, y, path, max) in enumerate(val_dataloader):
        path = path[0]
        ymax = max.item()
        assert x.shape[0] == 1, 'batch size must be 1 not {}'.format(x.shape[0])
        x = x.to(opt.device)
        y = y.to(opt.device)
        yhat = model(x)
        y = y.cpu().numpy().reshape((opt.n, opt.n))
        y = np.load(os.path.join(path, 'y.npy')) # TODO
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt':
            yhat = np.argmax(yhat, axis = 1)
        yhat = yhat.reshape((opt.n, opt.n))

        result_y = pca.fit(y)
        comp1_y = pca.components_[0]
        sign1_y = np.sign(comp1_y)

        result_yhat = pca.fit(yhat)
        comp1_yhat = pca.components_[0]
        sign1_yhat = np.sign(comp1_yhat)
        acc = np.sum((sign1_yhat == sign1_y)) / sign1_y.size
        acc_arr[i] = max(acc, 1 - acc)

        corr, pval = spearmanr(comp1_yhat, comp1_y)
        rho_arr[i] = abs(corr)

        corr, pval = pearsonr(comp1_yhat, comp1_y)
        p_arr[i] = abs(corr)

    print('PCA results:')
    print('Accuracy: {} +- {}'.format(np.mean(acc_arr), np.std(acc_arr)))
    print('Spearman R: {} +- {}'.format(np.mean(rho_arr), np.std(rho_arr)))
    print('Pearson R: {} +- {}'.format(np.mean(p_arr), np.std(p_arr)))
    print()

def argparseSetup():
    """Helper function to get default command line argument parser."""
    parser = argparse.ArgumentParser(description='Base parser')

    # pre-processing args
    parser.add_argument('--data_folder', type=str, default='test', help='Location of data')
    parser.add_argument('--toxx', type=str2bool, default=False, help='True if x should be converted to 2D image')
    parser.add_argument('--y_preprocessing', type=str, default='diag', help='type of pre-processing for y')
    parser.add_argument('--y_norm', type=str, default='batch', help='type of [0,1] normalization for y')
    parser.add_argument('--x_reshape', type=str2bool, default=True, help='True if x should be considered a 1D image')
    parser.add_argument('--ydtype', type=str2dtype, default='float32', help='torch data type for y')
    parser.add_argument('--y_reshape', type=str2bool, default=True, help='True if y should be considered a 2D image')
    parser.add_argument('--crop', type=str2list, help='size of crop to apply to image - format: <leftcrop-rightcrop>')

    parser.add_argument('--classes', type=int, default=10, help='number of classes in percentile normalization')

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
    parser.add_argument('--milestones', type=str2list, default = [2], help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr decay')
    parser.add_argument('--loss', type=str, default='mse', help='Type of loss to use: options: {"mse", "cross_entropy"}')

    # model args
    parser.add_argument('--model_type', type=str, help='Type of model')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='True if using a pretrained model')
    parser.add_argument('--resume_training', type=str2bool, default=False, help='True if resuming traning of a partially trained model')
    parser.add_argument('--ifile_folder', type=str, default='models/', help='Location of input file for pretrained model')
    parser.add_argument('--ifile', type=str, help='Name of input file for pretrained model')
    parser.add_argument('--ofile_folder', type=str, default='models/', help='Location to save checkpoint models')
    parser.add_argument('--ofile', type=str, default='model', help='Name of save file')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')

    # post-processing args
    parser.add_argument('--plot', type=str2bool, default=False, help='True to plot predictions') # TODO use this

    # SimpleEpiNet args
    parser.add_argument('--kernel_w_list', type=str2list, default=[5,5], help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=str2list, default=[10,10], help='List of hidden sizes for convolutional layers')

    # UNet args
    parser.add_argument('--nf', type=int, default=8, help='Number of filters')

    # DeepC args
    parser.add_argument('--dilation_list', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers')
    parser.add_argument('--hidden_size_dilation', type=int, default=10, help='Hidden size of dilated convolutional layers')

    # Akita args
    parser.add_argument('--dilation_list_trunk', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers of trunk')
    parser.add_argument('--bottleneck', type=int, default=10, help='Number of filters in bottleneck (must be <= hidden_size_dilation_trunk)')
    parser.add_argument('--dilation_list_head', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers of head')


    opt = parser.parse_args()
    # configure cuda
    if opt.gpus > 1:
        opt.cuda = True
        opt.use_parallel = True
        opt.gpu_ids = []
        for ii in range(6):
            try:
                torch.cuda.get_device_properties(ii)
                print(str(ii))
                opt.gpu_ids.append(ii)
            except AssertionError:
                print('Not ' + str(ii) + "!")
    elif opt.gpus == 1:
        opt.cuda = True
        opt.use_parallel = False
    else:
        opt.cuda = False
        opt.use_parallel = False

    if opt.cuda and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        opt.cuda = False
        opt.use_parallel = False

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt

def roundUpBy10(val):
    """Rounds value up to the nearst multiple of 10."""
    assert val > 0, "val too small"
    assert val < 10**10, "val too big"
    mult = 1
    while val > mult:
        mult *= 10
    return mult

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

def str2list(v):
    """
    Helper function for argparser, converts str to list by splitting on '-': "i-j-k" -> [i,j,k].

    Inputs:
        v: string
    """
    if v is None:
        return None
    if isinstance(v, str):
       return [int(i) for i in v.split('-')]
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
            print('Unkown str: {}'.format(v))
    else:
        raise argparse.ArgumentTypeError('str value expected.')

def list2str(v):
    """
    Helper function to undo str2list.

    Inputs:
        v: list
    """
    assert type(v) == list
    return '-'.join([str(i) for i in v])

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
    # plotPerClassAccuracy(None, None, 5)

if __name__ == '__main__':
    main()
