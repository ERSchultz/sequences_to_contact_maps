import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

from plotting_functions import plotContactMap

sys.path.insert(1, '/home/eric/TICG-chromatin/scripts')
sys.path.insert(1, 'C:\\Users\\Eric\\OneDrive\\Documents\\Research\\Coding\\TICG-chromatin\\scripts')
from get_seq import relabel_seq

def getArgs(dataset = None, model_id = None):
    parser = argparse.ArgumentParser(description='Base parser')
    # parser.add_argument('--root', type=str, default='C:\\Users\\Eric\\OneDrive\\Documents\\Research\\Coding\\sequences_to_contact_maps')
    parser.add_argument('--root', type=str, default='/home/eric/sequences_to_contact_maps')
    parser.add_argument('--dataset', type=str, default=dataset, help='Location of input data')
    parser.add_argument('--sample', type=int, default=40)
    parser.add_argument('--model_id', type=int, default=model_id)

    args = parser.parse_args()
    args.data_folder = osp.join(args.root, args.dataset)
    args.sample_folder = osp.join(args.data_folder, 'samples/sample{}'.format(args.sample))

    # check that model_id matches
    argparse_path = osp.join(args.root, 'results/ContactGNNEnergy/{}/argparse.txt'.format(args.model_id))
    with open(argparse_path, 'r') as f:
        for line in f:
            if line == '--data_folder\n':
                break
        dataset = f.readline().strip()
        dataset = osp.split(dataset)[1]
        assert dataset == args.dataset, 'Dataset mismatch: {} vs {}'.format(dataset, args.dataset)

    # create output directory
    args.odir = osp.join(args.sample_folder, str(args.model_id))
    if not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)

    # create logfile
    log_file_path = osp.join(args.odir, 'results_summary.log')
    args.log_file = open(log_file_path, 'w')

    return args

def get_x_s(args, plot=True):
    x = np.load(osp.join(args.sample_folder, 'x.npy'))
    x = x.astype(np.float64)
    m, k = x.shape

    if plot:
        for i in range(k):
            plt.plot(x[:, i])
            plt.title(r'$\Psi$[:, {}]'.format(i))
            plt.savefig(osp.join(args.sample_folder, 'x_{}'.format(i)))
            plt.close()

    s_file = osp.join(args.sample_folder, 's.npy')
    if osp.exists(s_file):
        s = np.load(s_file)
    else:
        chi_path1 = osp.join(args.data_folder, 'chis.npy')
        chi_path2 = osp.join(args.sample_folder, 'chis.npy')
        if osp.exists(chi_path1):
            chi = np.load(chi_path1)
        elif osp.exists(chi_path2):
            chi = np.load(chi_path2)
        else:
            raise Exception('chi not found at {} or {}'.format(chi_path1, chi_path2))
        chi = chi.astype(np.float64)
        print('Chi:\n', chi, file = args.log_file)

        s = x @ chi @ x.T

    return x, s

def get_contact(args):
    y = np.load(osp.join(args.sample_folder, 'y.npy'))

    ydiag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))

    return y, ydiag

def plot_top_PCs(inp, inp_type, odir, log_file, count = 1):
    '''
    Plots top PCs of inp.

    Inputs:
        inp: np array containing input data
        inp_type: str representing type of input data
        odir: output directory to save plots to
        log_file: output file to write results to
        count: number of PCs to plot

    Outputs:
        pca.components_: all PCs of inp
    '''

    pca = PCA()
    pca = pca.fit(inp)

    print("Explained ratio: ", np.round(pca.explained_variance_ratio_[0:4], 3), file = log_file)
    print("Singular values: ", np.round(pca.singular_values_[0:4], 3), file = log_file)

    i = 0
    tot_explained = 0
    while i < count:
        explained = pca.explained_variance_ratio_[i]
        tot_explained += explained
        plt.plot(pca.components_[i])
        plt.title("Component {}: {}% of variance".format(i+1, np.round(explained * 100, 3)))
        plt.savefig(osp.join(odir, '{}_PC_{}.png'.format(inp_type, i+1)))
        plt.close()
        i += 1

    plt.show()

    return pca.components_

def pearsonround(x, y):
    "Wrapper function that combines np.round and pearsonr."
    stat, _ = pearsonr(x, y)
    return np.round(stat, 2)

def find_linear_combinations(x, args, PC, verbose = False):
    m, k = x.shape
    for j in range(2):
        print('PC {}'.format(j+1), file = args.log_file)

        # correlations
        for i in range(k):
            stat = pearsonround(x[:, i], PC[j])
            print('\tCorrelation with particle type {}: {}'.format(i, stat), file = args.log_file)

        # linear regression
        if verbose:
            x2 = sm.add_constant(x)
            est = sm.OLS(PC[j], x2)
            est2 = est.fit()
            print(est2.summary(), '\n', file = args.log_file)
        else:
            reg = LinearRegression()
            reg.fit(x, PC[j])
            score = reg.score(x, PC[j])
            print('\n\tLinear Regression', file = args.log_file)
            print(f'\tR^2: {score}', file = args.log_file)
            print(f'\tcoefficients: {reg.coef_}', file = args.log_file)

def main(dataset, model_id):
    args = getArgs(dataset, model_id)
    print(args)

    x, e = get_x_s(args)
    y, ydiag = get_contact(args)

    ehat = np.loadtxt(osp.join(args.root, 'results/ContactGNNEnergy/{}/sample{}/energy_hat.txt'.format(args.model_id, args.sample)))
    mse = np.round(mean_squared_error(e, ehat), 3)
    plotContactMap(ehat, vmin = 'min', vmax = 'max', cmap = 'blue-red', ofile = osp.join(args.odir, 's_hat.png'), title = 'Model ID = {}\n {} (MSE Loss = {})'.format(args.model_id, r'$\hat{S}$', mse))

    dif = ehat - e
    v_max = np.max(e)
    plotContactMap(dif, osp.join(args.odir, 's_dif.png'), vmin = -1 * v_max, vmax = v_max, title = r'$\hat{S}$ - S', cmap = 'blue-red')

    print("\nY_diag", file = args.log_file)
    PC_y = plot_top_PCs(ydiag, 'y_diag', args.sample_folder, args.log_file, count = 2)

    print("\nS", file = args.log_file)
    print(f'Rank: {np.linalg.matrix_rank(e)}', file = args.log_file)
    PC_e = plot_top_PCs(e, 's', args.sample_folder, args.log_file, count = 2)
    stat = pearsonround(PC_y[0], PC_e[0])
    print("Correlation between PC 1 of y_diag and S: ", stat, file = args.log_file)

    print("\nS_hat", file = args.log_file)
    PC_ehat = plot_top_PCs(ehat, 's_hat', args.odir, args.log_file, count = 2)

    stat = pearsonround(PC_y[0], PC_ehat[0])
    print("Correlation between PC 1 of y_diag and S_hat: ", stat, file = args.log_file)
    for zero_index, one_index in enumerate([1,2,3]):
        stat = pearsonround(PC_e[zero_index], PC_ehat[zero_index])
        print(f"Correlation between PC {one_index} of S and S_hat: ", stat, file = args.log_file)

    # correlation between S and x
    print("\nS_hat vs X", file = args.log_file)
    find_linear_combinations(x, args, PC_e)

    print('\nNon-linear system', file = args.log_file)
    x = relabel_seq(x, 'D-AB')
    find_linear_combinations(x, args, PC_e)

    # for old in ['AB', 'BC', 'AC']:
    #     print("\nWhat if '{}' -> 'D'".format(old), file = args.log_file)
    #     x_new = relabel_seq(x_relabel, '{}-D'.format(old))
    #     find_linear_combinations(x_new, args, PC_e)

    print('\nAll possible pairwise interactions', file = args.log_file)
    m, k = x.shape
    k_new = int(k*(k+1)/2)
    x_new = np.zeros((m, k_new))
    ind = np.triu_indices(k)
    for i in range(m):
        x_new[i] = np.outer(x[i], x[i])[ind]

    find_linear_combinations(x_new, args, PC_e)


    rows = int(m*(m+1)/2)
    k_newer = int(k_new*(k_new+1)/2)
    X = np.zeros((rows, k_newer))
    Y = np.zeros(rows)
    ind = np.triu_indices(k_new)
    row = 0
    for i in range(m):
        for j in range(i, m):
            outer = np.outer(x_new[i], x_new[j])
            X[row] = outer[ind]

            Y[row] = e[i,j]
            row += 1
    print(Y.shape)
    print(X.shape)

    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    print(est2.summary(), '\n', file = args.log_file)

if __name__ == '__main__':
    for id in [42]:
        main('dataset_11_03_21', id)
