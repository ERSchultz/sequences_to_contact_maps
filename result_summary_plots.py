'''Mostly deprecated'''

import argparse
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_L
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import LETTERS, pearson_round
from scipy.sparse.csgraph import laplacian
# from scripts.argparse_utils import ArgparserConverter
# from scripts.load_utils import (get_final_max_ent_folder, load_all,
#                                load_max_ent_L)
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--root', type=str, default='/home/erschultz')
    parser.add_argument('--dataset', type=str,
                    help='Location of input data')
    parser.add_argument('--sample', type=AC.str2int, default=40)
    parser.add_argument('--sample_folder', type=AC.str2None, default=None,
                    help='None to infer from --dataset and --sample')
    parser.add_argument('--method', type=AC.str2None, default='GNN',
                    help = 'parametrization method')
    parser.add_argument('--model_id', type=AC.str2int,
                    help='model ID if method == GNN')
    parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha for linear model')
    parser.add_argument('--k', type=AC.str2int, help='k for method')
    parser.add_argument('--plot', type=AC.str2bool, default=False,
                    help='True to plot s_hat and s_dif')
    parser.add_argument('--experimental', type=AC.str2bool, default=False,
                        help='True if using experimental data (ground truth data missing)')
    parser.add_argument('--overwrite', type=AC.str2bool, default=False,
                    help='True to overwrite existing results')
    parser.add_argument('--robust', type=AC.str2bool, default=False,
                    help='True for robust PCA analysis')
    parser.add_argument('--scale', type=AC.str2bool, default=False,
                    help='True to scale data for PCA')
    parser.add_argument('--svd', type=AC.str2bool, default=False,
                    help='True to use svd instead of PCA')

    args = parser.parse_args()
    args.data_folder = osp.join(args.root, args.dataset)
    if args.sample_folder is None:
        assert args.sample is not None
        args.sample_folder = osp.join(args.data_folder, f'samples/sample{args.sample}')

    # determine what to plot
    if args.method is None:
        args.plot_baseline = True
        args.plot = False # override
        args.verbose = True
    else:
        args.plot_baseline = False
        args.verbose = False

    # check that model_id matches
    if args.model_id is not None:
        argparse_path = osp.join(args.root, f'results/ContactGNNEnergy/{args.model_id}/argparse.txt')
        with open(argparse_path, 'r') as f:
            for line in f:
                if line == '--data_folder\n':
                    break
            dataset = f.readline().strip()
            dataset = osp.split(dataset)[1]
            assert dataset == args.dataset, f'Dataset mismatch: {dataset} vs {args.dataset}'

    # create output directory
    if args.method is None:
        args.odir = osp.join(args.sample_folder, 'PCA_analysis')
    elif args.method == 'GNN':
        args.odir = osp.join(args.sample_folder, f'results_GNN-{args.model_id}')
    else:
        args.odir = osp.join(args.sample_folder, f'results_{args.method}-{args.k}')
    if args.scale:
        args.odir += '-scale'
    if args.svd:
        args.odir += '-svd'
    if not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)

    # create logfile
    log_file_path = osp.join(args.odir, 'results_summary.log')
    args.log_file = open(log_file_path, 'w')

    return args

def plot_top_PCs(inp, inp_type='', odir=None, log_file=sys.stdout, count=2,
                plot=False, verbose=False, scale=False, svd=False):
    '''
    Plots top PCs of inp.
    Inputs:
        inp: np array containing input data
        inp_type: str representing type of input data
        odir: output directory to save plots to
        log_file: output file to write results to
        count: number of PCs to plot
        plot: True to plot
        verbose: True to print
        scale: True to scale data before PCA
        svd: True to use right singular vectors instead of PCs
    Outputs:
        pca.components_: all PCs of inp
    '''
    pca = PCA()
    if svd:
        pca = None
        assert not scale
        U, S, Vt = np.linalg.svd(np.corrcoef(inp), full_matrices=0)
    else:
        if scale:
            try:
                pca = pca.fit(inp/np.std(inp, axis = 0))
            except ValueError:
                print(f'val error for {inp_type}')
                pca = pca.fit(inp)
        else:
            pca = pca.fit(inp)

        # combine notation between svd and pca
        S = pca.singular_values_
        Vt = pca.components_

    if verbose:
        if log_file is not None:
            print(f'\n{inp_type.upper()}', file = log_file)
            if pca is not None:
                print(f'''% of total variance explained for first {count} PCs:
                    {np.round(pca.explained_variance_ratio_[0:count], 3)}
                    \n\tSum of first {count}: {np.sum(pca.explained_variance_ratio_[0:count])}''',
                    file = log_file)
            print(f'''Singular values for first {count} PCs:
                {np.round(S[0:count], 3)}
                \n\tSum of all: {np.sum(S)}''',
                file = log_file)

    if plot:
        i = 0
        while i < count:
            PC = Vt[i]
            PC *= np.sign(PC[:100])
            # PCs are sign invariant, so this doesn't matter mathematically
            # goal is to help compare PCs visually by aligning them

            plt.plot(PC)
            if pca is not None:
                explained = pca.explained_variance_ratio_[i]
                plt.title(f"Component {i+1}: {np.round(explained * 100, 3)}% of variance")
            if odir is not None:
                plt.savefig(osp.join(odir, f'{inp_type}_PC_{i+1}.png'))
                plt.close()
            else:
                plt.show()
            i += 1

    return Vt

def project_L_to_psi_basis(L, psi):
    '''
    Projects energy into basis of psi.

    Inputs:
        L: L matrix (symmetric or asymmetric L)
        psi: bead labels

    Outputs:
        L_proj: L matrix in basis of psi
    '''
    hat_chi = predict_chi_in_psi_basis(psi, L)
    L_proj = calculate_L(psi, chi)
    return L_proj

def run_regression(X, Y, k_new, args, verbose = True):
    '''
    Linear regression to estimate chi given psi and Y (Y = X @ chi @ X.T).

    Inputs:
        X: bead labels, reformatted to allow for linear regression
        Y: contact map, reformated to allow for linear regression
        k_new: number of bead label pairs
        args: command line arguments (None overrides verbose)
        verbose: True to print

    Outputs:
        hat_chi: estimated chi matrix of shape (k_new, k_new)
    '''
    est = sm.OLS(Y, X)
    est = est.fit()
    if verbose:
        if args is not None:
            print(est.summary(), '\n', file = args.log_file)
        else:
            print(est.summary(), '\n')

    # construct chi
    # converts chi back to matrix format
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

    return hat_chi

def x_to_psi(x, mode = 'all_pairs'):
    '''
    Relabels x to contains all pairs of bead types.

    E.g. if x[i, :] = [A, B, C] where A, B, C are binary numbers
    psi[i, :] := [A, AB, AC, B, BC, C]
    ['A', 'AB', 'AC', 'B', 'BC', 'C'] will be returned as psi_letters

    Inputs:
        x: binary epigenetic mark array
        mode: mapping strategy (only all_pairs supported)
    Outputs:
        psi: relabled bead type array
        psi_letters: bead type labels corresponding to new bead types (see example)
    '''
    assert np.count_nonzero((x != 0) & (x != 1)) == 0, "x must be binary"
    assert mode == 'all_pairs'
    m, k = x.shape
    ell = int(k*(k+1)/2) # number of pairs of marks (including self-self)
                         # (i.e. number of labels in psi)
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

def find_all_pairs(psi, energy, letters):
    '''
    Finds all pairs of of particles and emumerates the pairs of bead types for each pair.
    Simulataneously finds corresponding energy values.

    E.g. If x[i,:] = [A_i, AB_i, B_i] and x[j,:] = [A_j, AB_j, B_j],
    then X[row, :] = [A_i*A_j, A_i*AB_j, A_i*B_j, AB_i*AB_j, AB_i*B_j, B_i*B_j]
    ['A-A', 'A-AB', 'A-B', 'AB-AB' 'AB-B', 'B-B'] will be returned as letters_new

    Inputs:
        psi: binary bead label array
        energy: matrix such that psi \chi psi^T = energy for some \chi
        letters: bead type labels corresponding to psi

    Outputs:
        X: x for linear regression
        Y: y for linear regression
        letters_new: bead type labels corresponding to new bead types (see example)
    '''
    m, k = psi.shape
    rows = int(m*(m+1)/2) # number of pairs of particles (including self-self)
    k_new = int(k*(k+1)/2) # number of pairs of bead types (including self-self)
    X = np.zeros((rows, k_new))
    Y = np.zeros(rows)
    ind = np.triu_indices(k)
    row = 0
    for i in range(m):
        for j in range(i, m):
            # get all pairs of marks
            outer = np.outer(psi[i], psi[j])
            outer = np.triu(outer) + np.triu(outer.T, 1)
            outer = outer[ind] # flatten
            X[row] = outer

            # get corresponding energy
            Y[row] = energy[i,j]
            row += 1

    # determine letters for X
    if letters is not None:
        letters_outer = np.empty((k, k), dtype=np.dtype('U5'))
        for i in range(k):
            for j in range(k):
                letters_outer[i,j] = letters[i] + '-' + letters[j]
        letters_new = letters_outer[ind]
        print('letters 2', letters_new)
    else:
        letters_new = None

    return X, Y, letters_new

def predict_chi_in_psi_basis(psi, L, psi_letters=None, args=None, verbose=False):
    '''
    Wrapper function to predict hat_chi given psi.
    '''
    if psi.shape[1] > psi.shape[0]:
        psi = psi.T # psi is m x k

    _, k = psi.shape

    # ensure symmetric L
    L_sym = (L + L.T)/2

    # find pairs of rows in psi
    X, Y, letters_newer = find_all_pairs(psi, L_sym, psi_letters)

    # run linear regression
    hat_chi = run_regression(X, Y, k, args, verbose = verbose)

    # save results
    if args is not None:
        print(np.round(hat_chi, 2), '\n', file = args.log_file)
        np.save(osp.join(args.odir, args.save_file), hat_chi)

    return hat_chi

def load_method_L(root, sample_folder, sample, method, k, model_id):
    if method == 'GNN':
        # look in results
        raise Exception('deprecated')
        gnn_path = osp.join(root, f'results/ContactGNNEnergy/{model_id}/sample{sample}/energy_hat.txt')
        if osp.exists(gnn_path):
            return np.loadtxt(gnn_path)
        else:
            k = 'knone'

    # Not GNN or gnn_path not found
    for method_file in os.listdir(sample_folder):
        method_path = osp.join(sample_folder, method)
        if method_file == method and osp.isdir(method_path):
            for k_file in os.listdir(method_path):
                k_path = osp.join(method_path, k_file)
                if int(k_file[1:]) == k and osp.isdir(k_path):
                    for replicate_file in os.listdir(k_path):
                        replicate_path = osp.join(k_path, replicate_file)
                        if osp.isdir(replicate_path):
                            L_hat = load_max_ent_L(k, replicate_path)
                            return L_hat # return first replicate

    raise Exception('L_hat not found')

def pca_analysis(args, y, ydiag, L, L_hat):
    # calculate PCA
    PC_y = plot_top_PCs(y, 'y', args.odir, args.log_file, count = 2,
                        plot = args.plot_baseline, verbose = args.verbose,
                        scale = args.scale, svd = args.svd)
    PC_y_diag = plot_top_PCs(ydiag, 'y_diag', args.odir, args.log_file, count = 6,
                        plot = args.plot_baseline, verbose = args.verbose,
                        scale = args.scale, svd = args.svd)


    y_log = np.log(y + 1)
    plot_matrix(y_log, osp.join(args.odir, 'y_log.png'), vmax = 'mean')
    PC_y_log = plot_top_PCs(y_log, 'y_log', args.odir, args.log_file, count = 2,
                        plot = args.plot_baseline, verbose = args.verbose,
                        scale = args.scale, svd = args.svd)

    meanDistLog = DiagonalPreprocessing.genomic_distance_statistics(y_log)
    y_log_diag = DiagonalPreprocessing.process(y_log, meanDistLog)
    plot_matrix(y_log_diag, osp.join(args.odir, 'y_log_diag.png'), vmin = 'center1',
                cmap='blue-red')
    PC_y_log_diag = plot_top_PCs(y_log_diag, 'y_log_diag', args.odir, args.log_file, count = 3,
                                plot = args.plot_baseline, verbose = args.verbose,
                                scale = args.scale, svd = args.svd)

    p = y/np.mean(np.diagonal(y))
    PC_p = plot_top_PCs(p, 'p', args.odir, args.log_file, count = 3,
                        plot = args.plot_baseline, verbose = args.verbose,
                        scale = args.scale, svd = args.svd)

    y_loginf = np.log(y)
    y_loginf[np.isinf(y_loginf)] = np.nan
    plot_matrix(y_loginf, osp.join(args.odir, 'y_loginf.png'), vmin = 'min', vmax = 'max')
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_loginf)
    y_loginf_diag = DiagonalPreprocessing.process(y_loginf, meanDist, verbose = False)
    np.nan_to_num(y_loginf_diag, copy = False, nan = 1.0)
    plot_matrix(y_loginf_diag, osp.join(args.odir, 'y_loginf_diag.png'),
                vmin = 'center1', cmap='blue-red')

    PC_y_loginf_diag = plot_top_PCs(y_loginf_diag, 'y_loginf_diag', args.odir,
                                args.log_file, count = 3,
                                plot = args.plot_baseline, verbose = args.verbose,
                                scale = args.scale, svd = args.svd)



    stat = pearson_round(PC_y[0], PC_y_log[0])
    print("Correlation between PC 1 of y and y_log: ", stat, file = args.log_file)
    stat = pearson_round(PC_y_diag[0], PC_y_log[0])
    print("Correlation between PC 1 of y_diag and y_log: ", stat, file = args.log_file)
    stat = pearson_round(PC_y_diag[1], PC_y_log[1])
    print("Correlation between PC 2 of y_diag and y_log: ", stat, file = args.log_file)

    # robust PCA
    if args.robust:
        raise Exception('deprecated')
        # L_log_file = osp.join(args.odir, 'L_log.npy')
        # L_file = osp.join(args.odir, 'L.npy')
        # S_log_file = osp.join(args.odir, 'S_log.npy')
        # S_file = osp.join(args.odir, 'S.npy')
        # if osp.exists(L_log_file) and not args.overwrite:
        #     L_log = np.load(L_log_file)
        #     L = np.load(L_file)
        #     S_log = np.load(S_log_file)
        #     S = np.load(S_file)
        # else:
        #     y_log = np.log(y + 1e-8)
        #     L_log, S_log = R_pca(y_log).fit(max_iter=2000)
        #     np.save(L_log_file, L_log)
        #     np.save(S_log_file, S_log)
        #     L = np.exp(L_log)
        #     S = np.exp(S_log)
        #     np.save(L_file, L)
        #     np.save(S_file, S)
        #
        # plot_matrix(L_log, osp.join(args.odir, 'L_log.png'), vmin = 'min', vmax = 'max', title = 'L_log')
        # plot_matrix(L, osp.join(args.odir, 'L.png'), vmin = 'min', vmax = 'max', title = 'L')
        # plot_matrix(S_log, osp.join(args.odir, 'S_log.png'), vmin = 'min', vmax = 'max', title = 'S_log')
        # plot_matrix(S, osp.join(args.odir, 'S.png'), vmin = 'min', vmax = 'max', title = 'S')
        #
        #
        # PC_L_log = plot_top_PCs(L_log, 'L_log', args.odir, args.log_file, count = 2,
        #                 plot = args.plot_baseline, verbose = args.verbose,
        #                 scale = args.scale, svd = args.svd)
        # stat = pearson_round(=PC_L_log[0], PC_y[0])
        # print("Correlation between PC 1 of L_log and Y: ", stat, file = args.log_file)
        # stat = pearson_round(=PC_L_log[1], PC_y[1])
        # print("Correlation between PC 2 of L_log and Y: ", stat, file = args.log_file)
        # stat = pearson_round(=PC_L_log[0], PC_y_diag[0])
        # print("Correlation between PC 1 of L_log and Y_diag: ", stat, file = args.log_file)
        # stat = pearson_round(=PC_L_log[1], PC_y_diag[1])
        # print("Correlation between PC 2 of L_log and Y_diag: ", stat, file = args.log_file)


    if args.method is None and args.overwrite:
        ## Plot projection of y in lower rank space
        for i in [1,2,5,10,15,100]:
            # get y top i PCs
            pca = PCA(n_components = i)
            if args.scale:
                y_transform = pca.fit_transform(y/np.std(y, axis = 0))
            else:
                y_transform = pca.fit_transform(y)
            y_i = pca.inverse_transform(y_transform)
            plot_matrix(y_i, osp.join(args.odir, f'y_rank_{i}.png'), vmax = 'max', title = f'Y rank {i}')

        ## Plot projection of y_diag in lower rank space
        for i in [1,2,5,10,15,100]:
            # get y_diag top i PCs
            pca = PCA(n_components = i)
            if args.scale:
                y_transform = pca.fit_transform(ydiag/np.std(ydiag, axis = 0))
            else:
                y_transform = pca.fit_transform(ydiag)
            y_i = pca.inverse_transform(y_transform)
            plot_matrix(y_i, osp.join(args.odir, f'y_diag_rank_{i}.png'), vmax = 'max', title = f'Y_diag rank {i}')
            if i == 1:
                np.save(osp.join(args.odir, f'y_diag_rank_{i}.npy'), y_i)


    if L is not None:
        L = (L + L.T) / 2
        PC_L = plot_top_PCs(L, 'L', args.odir, args.log_file, count = 2,
                        plot = args.plot_baseline, verbose = args.verbose,
                        scale = args.scale, svd = args.svd)
        print(f'Rank of L: {np.linalg.matrix_rank(L)}', file = args.log_file)
        if args.method is None:
            for i in range(4):
                stat = pearson_round(PC_y[i], PC_L[i])
                print(f"Correlation between PC {i+1} of y and L: ", stat, file = args.log_file)

            for i in range(4):
                stat = pearson_round(PC_y_diag[i], PC_L[i])
                print(f"Correlation between PC {i+1} of y_diag and L: ", stat, file = args.log_file)

            for i in range(4):
                stat = pearson_round(PC_y_log_diag[i], PC_L[i])
                print(f"Correlation between PC {i+1} of y_log_diag and L: ", stat, file = args.log_file)

            for i in range(4):
                stat = pearson_round(PC_y_loginf_diag[i], PC_L[i])
                print(f"Correlation between PC {i+1} of y_log_inf_diag and L: ", stat, file = args.log_file)

    if L_hat is not None:
        PC_L_hat = plot_top_PCs(L_hat, 'L_hat', args.odir, args.log_file, count = 2,
                        plot = True, verbose = True, scale = args.scale, svd = args.svd)

        stat = pearson_round(PC_y_diag[0], PC_L_hat[0])
        print("Correlation between PC 1 of y_diag and L_hat: ", stat, file = args.log_file)
        for zero_index, one_index in enumerate([1,2,3]):
            stat = pearson_round(PC_L[zero_index], PC_L_hat_hat[zero_index])
            print(f"Correlation between PC {one_index} of L and L_hat: ", stat, file = args.log_file)

def main():
    args = getArgs()
    print(args)

    ## load data ##
    x, psi, chi, chi_diag, L, y, ydiag = load_all(args.sample_folder, True,
                                                args.data_folder, args.log_file,
                                                experimental = args.experimental,
                                                throw_exception = False)

    if args.method is not None:
        L_hat = load_method_S(args.root, args.sample_folder, args.sample, args.method, args.k, args.model_id)
    else:
        L_hat = None

    # pairwise PC correlation
    pca_analysis(args, y, ydiag, L, L_hat)

    # ## Compare MSE in PCA space ##
    # for i in range(1, 4):
    #     # get e top i PCs
    #     pca = PCA(n_components = i)
    #     e_transform = pca.fit_transform(e)
    #     e_i = pca.inverse_transform(e_transform)
    #     np.save(osp.join(args.odir, f'e_{i}.npy'), e_i)
    #     plot_matrix(e_i, osp.join(args.odir, f'e_{i}.png'), vmin = 'min', vmax = 'max', cmap = 'blue-red', title = f"E rank {i}")
    #
    #
    #     # compare ehat to projection of e onto top PCs
    #     mse = np.round(mean_squared_error(e_i, e_hat), 3)
    #     print(f'E top {i} PCs - MSE: {mse}', file = args.log_file)
    #
    # ## plot e_hat and e_dif ##
    # mse_e = np.round(mean_squared_error(e, e_hat), 3)
    # mse_s = np.round(mean_squared_error(s, s_hat), 3)
    # print(f'E - MSE: {mse_e}', file = args.log_file)
    # print(f'S - MSE: {mse_s}', file = args.log_file)
    # if args.plot:
    #     # e
    #     if args.method == 'GNN':
    #         plot_title = 'Model ID = {}\n {} (MSE Loss = {})'.format(args.model_id, r'$\hat{E}$', mse_e)
    #     else:
    #         plot_title = '{} (MSE Loss = {})'.format(r'$\hat{E}$', mse_e)
    #     plot_matrix(e_hat, osp.join(args.odir, 'e_hat.png'), vmin = np.min(e), vmax = np.max(e), cmap = 'blue-red', title = plot_title)
    #
    #     dif = e_hat - e
    #     v_max = np.max(e)
    #     plot_matrix(dif, osp.join(args.odir, 'e_dif.png'), vmin = -1 * v_max, vmax = v_max, title = r'$\hat{E}$ - E', cmap = 'blue-red')
    #
    #     # s
    #     if args.method == 'GNN':
    #         plot_title = 'Model ID = {}\n {} (MSE Loss = {})'.format(args.model_id, r'$\hat{S}$', mse_s)
    #     else:
    #         plot_title = '{} (MSE Loss = {})'.format(r'$\hat{S}$', mse_s)
    #     plot_matrix(s_hat, osp.join(args.odir, 's_hat.png'), vmin = 'min', vmax = 'max', cmap = 'blue-red', title = plot_title)
    #
    #     dif = s_hat - s
    #     v_max = np.max(s)
    #     plot_matrix(dif, osp.join(args.odir, 's_dif.png'), vmin = -1 * v_max, vmax = v_max, title = r'$\hat{S}$ - S', cmap = 'blue-red')
    #
    #
    #
    # ## All pairs of bead types for all pairs of particles ##
    # print('\nAll possible pairwise interactions', file = args.log_file)
    # # first relabel marks with all possible pairs of marks for each bead
    # psi_tilde, psi_letters = x_to_psi(x)
    #
    # args.save_file = 'chi_hat.npy'
    # hat_chi = predict_chi_in_psi_basis(psi_tilde, s_hat, psi_letters, args)
    # if args.plot:
    #     s_hat_psi_basis = psi_tilde @ hat_chi @ psi_tilde.T
    #     e_hat_psi_basis = s_to_E(s_hat_psi_basis)
    #
    #     mse = np.round(mean_squared_error(e, e_hat_psi_basis), 3)
    #     if args.method == 'GNN':
    #         plot_title = 'Model ID = {}\n {} (MSE Loss = {})'.format(args.model_id, r'$\hat{E}_{\Psi-basis}$', mse)
    #     else:
    #         plot_title = '{} (MSE Loss = {})'.format(r'$\hat{E}_{\Psi-basis}$', mse)
    #     plot_matrix(e_hat_psi_basis, vmin = np.min(e), vmax = np.max(e), cmap = 'blue-red', ofile = osp.join(args.odir, 'e_hat_psi_basis.png'), title = plot_title)
    #
    # post_analysis_chi(args, psi_letters)

def post_analysis_chi(args, letters):
    chi = np.load(osp.join(args.sample_folder, 'chis.npy'))
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
    print(f'% of time sign matches: {np.round(sign_matches / possible_matches, 3)}',
        file = args.log_file)

    dif = chi_hat - chi
    mse = mean_squared_error(chi, chi_hat)
    print(f'MSE: {np.round(mse, 3)}', file = args.log_file)
    print(f'RMSE: {np.round(mse**0.5, 3)}', file = args.log_file)

    max = np.max(np.abs(chi))
    min = -1 * max
    plot_matrix(chi_hat, vmin=min, vmax=max, cmap='blue-red',
                ofile = osp.join(args.odir, 'chi_hat.png'), x_ticks = letters,
                y_ticks = letters)
    plot_matrix(chi, vmin=min, vmax=max, cmap='blue-red',
                ofile = osp.join(args.odir, 'chi.png'), x_ticks = letters,
                y_ticks = letters)
    plot_matrix(dif, vmin=min, vmax=max, cmap='blue-red',
                ofile = osp.join(args.odir, 'dif.png'), x_ticks = letters,
                y_ticks = letters)

def test_project():
    dir = '/home/erschultz/downsampling_analysis/samples3/sample100'
    max_ent_dir = osp.join(dir, 'optimize_grid_b_140_phi_0.03-max_ent10')
    final = get_final_max_ent_folder(max_ent_dir)
    L_pca = load_max_ent_L(final)

    x, psi, chi, chi_diag, L, y, ydiag = load_all(dir)
    x = x.T; psi = psi.T
    assert np.allclose(x, psi) # these should be the same

    # confirm that projecting s returns s
    L_proj = project_L_to_psi_basis(L, psi)
    assert np.allclose(L, L_proj)

    # visualize pca L projection
    L_pca_proj = project_L_to_psi_basis(L_pca, psi)
    plot_matrix(L_pca_proj, vmin = 'min', vmax = 'max', cmap = 'blue-red',
                ofile = osp.join(max_ent_dir, 'L_pca_proj.png'))
    plot_matrix(L_pca, vmin = 'min', vmax = 'max', cmap = 'blue-red',
                ofile = osp.join(max_ent_dir, 'L.png'))


    # # what if you project s into noise basis
    # m, k = psi.shape
    # s_noise_proj, e_noise_proj = project_L_to_psi_basis(s, np.random.rand(m,k))
    # # plot_matrix(s_noise_proj, vmin = 'min', vmax = 'max', cmap = 'blue-red',
    # #             ofile = osp.join(dir, 's_pca_proj.png'))
    # plot_matrix(e_noise_proj, vmin = 'min', vmax = 'max', cmap = 'blue-red',
    #             ofile = osp.join(dir, 'e_noise_proj.png'))
    # np.save(osp.join(dir, 'e_noise_proj.npy'), e_noise_proj)


if __name__ == '__main__':
    # main()
    test_project()
