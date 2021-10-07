import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy.stats import pearsonr

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from plotting_functions import plotContactMap

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # parser.add_argument('--root', type=str, default='C:\\Users\\Eric\\OneDrive\\Documents\\Research\\Coding\\sequences_to_contact_maps')
    parser.add_argument('--root', type=str, default='/home/eric/Research/sequences_to_contact_maps')
    parser.add_argument('--dataset', type=str, default='dataset_04_18_21', help='Location of input data')
    parser.add_argument('--sample', type=int, default=40)
    parser.add_argument('--model_id', type=int, default=26)

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

    # create logfile
    log_file_path = osp.join(args.sample_folder, 'results_summary.log')
    args.log_file = open(log_file_path, 'w')

    return args

def plot_x_chi(args, minus = False):
    x = np.load(osp.join(args.sample_folder, 'x.npy'))
    x = x.astype(np.float64)
    m, k = x.shape

    for i in range(k):
        plt.plot(x[:, i])
        plt.title(r'$\Psi$[:, {}]'.format(i))
        plt.savefig(osp.join(args.sample_folder, 'x_{}'.format(i)))
        plt.close()

    if minus:
        minus = x[:,1] - x[:,0]
        plt.plot(minus)
        plt.title(r'$\Psi$[:, 1] - $\Psi$[:, 0]')
        plt.savefig(osp.join(args.sample_folder, 'x_1-0'))
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

    if minus:
        return x, chi, minus
    else:
        return x, chi

def get_contact(args):
    y = np.load(osp.join(args.sample_folder, 'y.npy'))

    ydiag = np.load(osp.join(args.sample_folder, 'y_diag_instance.npy'))

    return y, ydiag

def plot_top_PCs(inp, args, inp_type, count = 1):
    pca = PCA()
    pca = pca.fit(inp)

    print("Explained ratio: ", np.round(pca.explained_variance_ratio_[0:4], 3), file = args.log_file)
    print("Singular values: ", np.round(pca.singular_values_[0:4], 3), file = args.log_file)

    i = 0
    tot_explained = 0
    while i < count:
        explained = pca.explained_variance_ratio_[i]
        tot_explained += explained
        plt.plot(pca.components_[i])
        plt.title("Component {}: {}% of variance".format(i+1, np.round(explained * 100, 3)))
        plt.savefig(osp.join(args.sample_folder, '{}_PC_{}.png'.format(inp_type, i+1)))
        plt.close()
        i += 1

    plt.show()

    return pca.components_[0]

def main():
    args = getArgs()

    x, chi = plot_x_chi(args)
    y, ydiag = get_contact(args)

    e = x @ chi @ x.T

    ehat = np.loadtxt(osp.join(args.root, 'results/ContactGNNEnergy/{}/sample{}/energy_hat.txt'.format(args.model_id, args.sample)))
    mse = np.round(mean_squared_error(e, ehat), 3)
    plotContactMap(ehat, vmin = 'min', vmax = 'max', cmap = 'blue-red', ofile = osp.join(args.sample_folder, 's_hat.png'), title = 'Model ID = {}\nMSE = {}'.format(args.model_id, mse))

    dif = ehat - e
    v_max = np.max(e)
    plotContactMap(dif, osp.join(args.sample_folder, 's_dif.png'), vmin = -1 * v_max, vmax = v_max, title = r'$\hat{S}$ - S', cmap = 'blue-red')

    print("\nY_diag", file = args.log_file)
    comp1y = plot_top_PCs(ydiag, args, 'y_diag', count = 2)

    print("\nS", file = args.log_file)
    comp1e = plot_top_PCs(e, args, 's', count = 2)
    stat, _ = pearsonr(comp1y, comp1e)
    print("Corrrelation between PC 1 of y_diag and S: ", stat, file = args.log_file)

    print("\nS_hat", file = args.log_file)
    comp1ehat = plot_top_PCs(ehat, args, 's_hat')

    stat, _ = pearsonr(comp1y, comp1ehat)
    print("Corrrelation between PC 1 of y_diag and S_hat: ", stat, file = args.log_file)
    stat, _ = pearsonr(comp1e, comp1ehat)
    print("Corrrelation between PC 1 of S and S_hat: ", stat, file = args.log_file)



if __name__ == '__main__':
    main()
