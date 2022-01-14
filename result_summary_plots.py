import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from scipy.stats import pearsonr
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from plotting_functions import plotContactMap

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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

def run_regression(X, Y, k_new, args, ofile, verbose = True):
    est = sm.OLS(Y, X)
    est = est.fit()
    if verbose:
        print(est.summary(), '\n', file = args.log_file)

    # construct chi
    hat_chi = np.zeros((k_new, k_new))
    row = 0
    col = 0
    for val in est.params:
        if row == col:
            hat_chi[row, col] = val
        else:
            hat_chi[row, col] = val * 2

        if col == k_new - 1:
            row += 1
            col = row
        else:
            col += 1

    print(np.round(hat_chi, 2), '\n', file = args.log_file)
    np.save(osp.join(args.odir, ofile + '.npy'), hat_chi)

def relabel_x(x):
    '''
    Relabels x to contains all pairs of bead types.

    E.g. if x[i, :] = [A, B, C] where A, B, C are binary numbers
    psi[i, :] := [A, AB, AC, B, BC, C]
    ['A', 'AB', 'AC', 'B', 'BC', 'C'] will be returned as psi_letters

    Inputs:
        x: binary bead type array
    Outputs:
        psi: relabled bead type array
        psi_letters: bead type labels corresponding to new bead types (see example)
    '''
    assert np.count_nonzero((x != 0) & (x != 1)) == 0, "x must be binary"
    m, k = x.shape
    ell = int(k*(k+1)/2) # number of pairs of marks (including self-self)
    psi = np.zeros((m, ell)) # contains all pairs of marks for each bead
    ind = np.triu_indices(k)
    for i in range(m):
        psi[i] = np.outer(x[i], x[i])[ind]

    # determine letters for x_new
    letters_outer = np.empty((k, k), dtype=np.dtype('U2'))
    for i in range(k):
        for j in range(k):
            if i == j:
                pair = LETTERS[i]
            else:
                pair = LETTERS[i] + LETTERS[j]
            letters_outer[i,j] = pair
    psi_letters = letters_outer[ind]
    print('letters 1', psi_letters)

    return psi, psi_letters

def find_all_pairs(x, energy, letters):
    '''
    Finds all pairs of of particles and emumerates the pairs of bead types for each pair.

    E.g. If x[i,:] = [A_i, AB_i, B_i] and x[j,:] = [A_j, AB_j, B_j],
    then X[row, :] = [A_i*A_j, A_i*AB_j, A_i*B_j, AB_i*AB_j, AB_i*B_j, B_i*B_j]
    ['A-A', 'A-AB', 'A-B', 'AB-AB' 'AB-B', 'B-B'] will be returned as letters_new

    Inputs:
        x: binary bead type array
        energy: matrix such that x \chi x^T = energy for some \chi
        letters: bead type labels corresponding to x

    Outputs:
        X: x for linear regression
        Y: y for linear regression
        letters_new: bead type labels corresponding to new bead types (see example)
    '''
    m, k = x.shape
    rows = int(m*(m+1)/2) # number of pairs of particles (including self-self)
    k_new = int(k*(k+1)/2) # number of pairs of bead types (including self-self)
    X = np.zeros((rows, k_new))
    Y = np.zeros(rows)
    ind = np.triu_indices(k)
    row = 0
    for i in range(m):
        for j in range(i, m):
            # get all pairs of marks
            outer = np.outer(x[i], x[j])
            outer = np.triu(outer) + np.triu(outer.T, 1)
            outer = outer[ind] # flatten
            X[row] = outer

            # get corresponding energy
            Y[row] = energy[i,j]
            row += 1

    # determine letters for X
    letters_outer = np.empty((k, k), dtype=np.dtype('U5'))
    for i in range(k):
        for j in range(k):
            letters_outer[i,j] = letters[i] + '-' + letters[j]
    letters_new = letters_outer[ind]
    print('letters 2', letters_new)

    return X, Y, letters_new

def regression_on_all_pairs(psi, psi_letters, chi, s, s_hat, args):
    _, ell = psi.shape

    for energy, text in zip([s, s_hat], ['chi', 'chi_hat']):
        X, Y, letters_newer = find_all_pairs(psi, (energy + energy.T)/2, psi_letters)

        # run linear regression
        run_regression(X, Y, ell, args, text)

def get_ground_truth(args, plot=True):
    x = np.load(osp.join(args.sample_folder, 'x.npy'))
    psi = np.load(osp.join(args.sample_folder, 'psi.npy'))
    x = x.astype(np.float64)
    m, k = x.shape

    if plot:
        for i in range(k):
            plt.plot(x[:, i])
            plt.title(r'$X$[:, {}]'.format(i))
            plt.savefig(osp.join(args.sample_folder, 'x_{}'.format(i)))
            plt.close()

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

    s_file = osp.join(args.sample_folder, 's.npy')
    if osp.exists(s_file):
        s = np.load(s_file)
    else:
        s = psi @ chi @ psi.T

    e_file = osp.join(args.sample_folder, 'e.npy')
    if osp.exists(e_file):
        e = np.load(e_file)
    else:
        e = s + s.T - np.diag(np.diagonal(s.copy()))

    y = np.load(osp.join(args.sample_folder, 'y.npy'))

    ydiag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))

    return x, chi, s, e, y, ydiag

def plot_top_PCs(inp, inp_type=None, odir = None, log_file = sys.stdout, count = 2, plot = False):
    '''
    Plots top PCs of inp.

    Inputs:
        inp: np array containing input data
        inp_type: str representing type of input data
        odir: output directory to save plots to
        log_file: output file to write results to
        count: number of PCs to plot
        plot: True to plot

    Outputs:
        pca.components_: all PCs of inp
    '''

    pca = PCA()
    pca = pca.fit(inp)

    print("% of total variance explained for first 4 PCs: ", np.round(pca.explained_variance_ratio_[0:4], 3), file = log_file)
    print("Singular values for first 4 PCs: ", np.round(pca.singular_values_[0:4], 3), file = log_file)

    if plot:
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
    # deprecated function to calculate PC from linear regression on x
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

def main(dataset, model_id, plot = True):
    args = getArgs(dataset, model_id)
    print(args)

    ## load data ##
    x, chi, s, e, y, ydiag = get_ground_truth(args)

    s_hat = np.loadtxt(osp.join(args.root, 'results/ContactGNNEnergy/{}/sample{}/energy_hat.txt'.format(args.model_id, args.sample)))

    ## plot s_hat and s_dif ##
    mse = np.round(mean_squared_error(s, s_hat), 3)
    if plot:
        plotContactMap(s_hat, vmin = 'min', vmax = 'max', cmap = 'blue-red', ofile = osp.join(args.odir, 's_hat.png'), title = 'Model ID = {}\n {} (MSE Loss = {})'.format(args.model_id, r'$\hat{S}$', mse))

        dif = s_hat - s
        v_max = np.max(s)
        plotContactMap(dif, osp.join(args.odir, 's_dif.png'), vmin = -1 * v_max, vmax = v_max, title = r'$\hat{S}$ - S', cmap = 'blue-red')

    # Compare PCs ##
    print("\nY_diag", file = args.log_file)
    PC_y = plot_top_PCs(ydiag, 'y_diag', args.sample_folder, args.log_file, count = 2, plot = plot)

    print("\nS", file = args.log_file)
    print(f'Rank: {np.linalg.matrix_rank(s)}', file = args.log_file)
    PC_s = plot_top_PCs(s, 's', args.sample_folder, args.log_file, count = 2, plot = plot)
    stat = pearsonround(PC_y[0], PC_s[0])
    print("Correlation between PC 1 of y_diag and S: ", stat, file = args.log_file)

    s_sym = (s + s.T)/2
    print("\nS_sym", file = args.log_file)
    print(f'Rank: {np.linalg.matrix_rank(s_sym)}', file = args.log_file)
    PC_s_sym = plot_top_PCs(s_sym, 's_sym', args.sample_folder, args.log_file, count = 2, plot = plot)
    stat = pearsonround(PC_y[0], PC_s_sym[0])
    print("Correlation between PC 1 of y_diag and S_sym: ", stat, file = args.log_file)
    stat = pearsonround(PC_y[1], PC_s_sym[1])
    print("Correlation between PC 2 of y_diag and S_sym: ", stat, file = args.log_file)

    # Compare MSE in PCA space ##
    print("\nS_hat", file = args.log_file)
    print(f'S - MSE: {mse}', file = args.log_file)
    for i in range(1, 4):
        # get e top i PCs
        pca = PCA(n_components = i)
        s_transform = pca.fit_transform(s)
        s_i = pca.inverse_transform(s_transform)

        # compare ehat to projection of e onto top PCs
        mse = np.round(mean_squared_error(s_i, s_hat), 3)
        print(f'S top {i} PCs - MSE: {mse}', file = args.log_file)
    PC_s_hat = plot_top_PCs(s_hat, 's_hat', args.odir, args.log_file, count = 2)

    # Compare y_diag and ehat ##
    stat = pearsonround(PC_y[0], PC_s_hat[0])
    print("Correlation between PC 1 of y_diag and S_hat: ", stat, file = args.log_file)
    for zero_index, one_index in enumerate([1,2,3]):
        stat = pearsonround(PC_s[zero_index], PC_s_hat[zero_index])
        print(f"Correlation between PC {one_index} of S and S_hat: ", stat, file = args.log_file)

    ## All pairs of bead types for all pairs of particles ##
    print('\nAll possible pairwise interactions', file = args.log_file)
    # first relabel marks with all possible pairs of marks for each bead
    psi_tilde, psi_letters = relabel_x(x)

    regression_on_all_pairs(psi_tilde, psi_letters, chi, s, s_hat, args)

    post_analysis_chi(args, psi_letters)

def post_analysis_chi(args, letters):
    chi = np.load(osp.join(args.odir, 'chi.npy'))
    k, _ = chi.shape
    chi_sign = np.zeros_like(chi)
    chi_sign[chi > 0] = 1
    chi_sign[chi < 0] = -1

    chi_hat = np.load(osp.join(args.odir, 'chi_hat.npy'))
    chi_hat_sign = np.zeros_like(chi_hat)
    chi_hat_sign[chi_hat > 0] = 1
    chi_hat_sign[chi_hat < 0] = -1

    sign_matches = np.sum(chi_hat_sign == chi_sign) # count number of times sign matches
    sign_matches -=  k * (k - 1) / 2 # subtract off lower diagonal
    possible_matches = k * (k + 1) / 2 # size of upper triangle
    print(sign_matches)
    print(f'% of time sign matches: {np.round(sign_matches / possible_matches, 3)}')

    dif = chi_hat - chi
    mse = mean_squared_error(chi, chi_hat)
    print(f'MSE: {np.round(mse, 3)}', file = args.log_file)
    print(f'RMSE: {np.round(mse**0.5, 3)}', file = args.log_file)

    plotContactMap(chi_hat, vmin=-2, vmax=2, cmap='blue-red', ofile = osp.join(args.odir, 'chi_hat.png'), x_ticks = letters, y_ticks = letters)
    plotContactMap(chi, vmin=-2, vmax=2, cmap='blue-red', ofile = osp.join(args.odir, 'chi.png'), x_ticks = letters, y_ticks = letters)
    plotContactMap(dif, vmin=-2, vmax=2, cmap='blue-red', ofile = osp.join(args.odir, 'dif.png'), x_ticks = letters, y_ticks = letters)



if __name__ == '__main__':
    for id in [42]:
        main('dataset_11_03_21', id, False)
