import argparse
import os
import os.path as osp
import time

import imageio
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from utils.load_utils import load_sc_contacts
from utils.plotting_utils import plot_matrix
from utils.utils import (diagonal_preprocessing_bulk,
                         genomic_distance_statistics, pearson_round,
                         print_size, print_time, triu_to_full)
from utils.xyz_utils import (find_dist_between_centroids, find_label_centroid,
                             lammps_load, xyz_load, xyz_to_contact_grid)

import dmaps  # https://github.com/hsidky/dmaps


def getArgs(default_dir='/home/erschultz/sequences_to_contact_maps/dataset_test/samples/sample92'):
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--dir', type=str, default=default_dir, help='location of data')
    parser.add_argument('--odir', type=str, help='location to write to')

    args = parser.parse_args()
    if args.odir is None:
        args.odir = osp.join(args.dir, 'diffusion_test')
    if not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)
    return args

def tune_epsilon(dmap, ofile):
    X = np.arange(-4, 12, 1)
    epsilons = np.exp(X)
    Y = np.zeros_like(epsilons)
    for i, eps in enumerate(epsilons):
        Y[i] = np.log(dmap.sum_similarity_matrix(eps))

    # find best linear regression on subset of 3/4 points
    best_indices = (-1, -1)
    best_score = 0
    best_reg = None
    for size in [3, 4, 5]:
        left = 0; right = size
        while right <= len(X):
            slice_X = X[left:right].reshape(-1, 1)
            slice_Y = Y[left:right]
            reg = LinearRegression().fit(slice_X, slice_Y)
            score = reg.score(slice_X, slice_Y)
            if score >= best_score and reg.coef_ > 0.1:
                best_score = score
                best_indices = (left, right)
                best_reg = reg
            left += 1
            right += 1

    # choose epsilon
    slice_X = X[best_indices[0]:best_indices[1]]
    slice_Y = Y[best_indices[0]:best_indices[1]]
    coef = np.round(best_reg.coef_[0], 3)
    intercept = np.round(best_reg.intercept_, 3)

    eps_final = np.round(np.exp(np.mean(slice_X)), 3)
    print(f'Using epsilon = {eps_final}')

    # plot results
    slice_X_big = X[best_indices[0]-1:best_indices[1]+1]
    plt.plot(X, Y)
    plt.plot(slice_X_big, slice_X_big * coef + intercept, color = 'k')
    plt.scatter(X, Y, facecolors='none', edgecolors='b')
    plt.scatter(slice_X, slice_Y)
    plt.text(np.min(X)+1, np.percentile(Y, 80), f'y = {coef}x + {intercept}'
                                                '\n'
                                                rf'$R^2$={np.round(best_score, 3)}')
    plt.xlabel(r'ln($\epsilon$)', fontsize = 16)
    plt.ylabel(r'ln$\sum_{i,j}A_{i,j}$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

    return eps_final

def plot_eigenvectors(v, xyz, odir, sc_contacts = None):
    N = len(v)

    # plot first 2 nonzero eigenvectors, color by order
    sc = plt.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_2$', fontsize = 16)
    plt.ylabel(r'$v_3$', fontsize = 16)
    plt.savefig(osp.join(odir, 'projection23.png'))
    plt.close()

    # plot 3 and 4 eigenvectors, color by order
    sc = plt.scatter(v[:,2]/v[:,0], v[:,3]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.savefig(osp.join(odir, 'projection34.png'))
    plt.close()

    # plot 2,3,4 eigenvectors, color by order
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], v[:,3]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    ax.set_xlabel(r'$v_2$', fontsize = 16)
    ax.set_ylabel(r'$v_3$', fontsize = 16)
    ax.set_zlabel(r'$v_4$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'projection234.png'))
    plt.close()

    # k_means
    num_vecs = 3
    X = v[:,1:1+num_vecs]
    sil_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k).fit(X)
        sil_scores.append(silhouette_score(X, kmeans.labels_))
    plt.plot(np.arange(2, 10, 1), sil_scores)
    plt.savefig(osp.join(odir, 'k_means_sil_score.png'))
    plt.close()
    k = np.argmax(sil_scores) + 2
    print(f'Using k = {k}')
    kmeans = KMeans(n_clusters=k).fit(X)

    # plot average contact map within cluster
    for cluster in range(k):
        ind = np.argwhere(kmeans.labels_ == cluster)
        if sc_contacts is not None:
            y_cluster = triu_to_full(np.sum(sc_contacts[ind.reshape(-1), :], axis = 0))
        else:
            xyz_ind = xyz[ind].reshape(len(ind), -1, 3)
            y_cluster = xyz_to_contact_grid(xyz_ind, 28.7)
        plot_matrix(y_cluster, osp.join(odir, f'cluster{cluster}_contacts.png'),
                    vmax = 'max', title = f'cluster {k}')

    # plot first 2 nonzero eigenvectors, color by kmeans
    cmap = matplotlib.cm.get_cmap('tab10')
    ind = np.arange(k) % cmap.N
    colors = plt.cycler('color', cmap(ind))
    for cluster, c in zip(range(k), colors):
        ind = np.argwhere(kmeans.labels_ == cluster)
        plt.scatter(v[ind, 1]/v[ind, 0], v[ind, 2]/v[ind, 0], color = c['color'],
                    label = cluster)
        plt.xlabel(r'$v_2$', fontsize = 16)
        plt.ylabel(r'$v_3$', fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'projection23_kmeans.png'))
    plt.close()

    # plot 3 and 4 eigenvectors, color by kmeans
    for cluster, c in zip(range(k), colors):
        ind = np.argwhere(kmeans.labels_ == cluster)
        plt.scatter(v[ind, 2]/v[ind, 0], v[ind, 3]/v[ind, 0], color = c['color'],
                    label = cluster)
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'projection34_kmeans.png'))
    plt.close()


    # plot 2,3,4 eigenvectors, color by kmeans
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for cluster, c in zip(range(k), colors):
        ind = np.argwhere(kmeans.labels_ == cluster)
        ax.scatter(v[ind,1]/v[ind,0], v[ind,2]/v[ind,0], v[ind,3]/v[ind,0], color = c['color'], label = cluster)
    ax.set_xlabel(r'$v_2$', fontsize = 16)
    ax.set_ylabel(r'$v_3$', fontsize = 16)
    ax.set_zlabel(r'$v_4$', fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'projection234_kmeans.png'))
    plt.close()

def main_xyz():
    args = getArgs()

    # load xyz
    t0 = time.time()
    xyz_file = osp.join(args.dir, 'data_out/output.xyz')
    lammps_file = osp.join(args.dir, 'traj.dump.lammpstrj')
    if osp.exists(xyz_file):
        xyz = xyz_load(xyz_file,
                    multiple_timesteps = True, save = True, N_min = 10,
                    N_max = None, down_sampling = 1)
    elif osp.exists(lammps_file):
        xyz = lammps_load(lammps_file, save = False, N_min = 2000,
                    N_max = None, down_sampling = 1)

    N, m, _ = xyz.shape
    xyz = xyz.reshape(N, m * 3)
    tf = time.time()
    print_time(t0, tf, 'load')

    # compute distance
    t0 = time.time()
    dist = dmaps.DistanceMatrix(xyz)
    dist.compute(metric=dmaps.metrics.rmsd)
    D = dist.get_distances()
    plot_matrix(D, ofile = osp.join(args.odir, 'distances.png'),
                    vmin = 'min', vmax = 'max')
    tf = time.time()
    print_time(t0, tf, 'distance')

    # compute eigenvectors
    dmap = dmaps.DiffusionMap(dist)
    eps = tune_epsilon(dmap, osp.join(args.odir, 'tuning.png'))
    dmap.set_kernel_bandwidth(eps)
    dmap.compute(5, 0.5)

    v = dmap.get_eigenvectors()
    w = dmap.get_eigenvalues()
    print('w', w)
    order = np.argsort(v[:, 1])
    print('\n\norder corr: ', pearson_round(order, np.arange(0, N, 1), stat = 'spearman'))

    plot_eigenvectors(v, xyz.reshape(-1, m, 3), args.odir)

def main_sc():
    dir = '/home/erschultz/dataset_test/samples/sample92'
    odir = osp.join(dir, 'single_cell_diffusion')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    overall = np.load(osp.join(dir, 'y.npy'))
    mean_per_diag = genomic_distance_statistics(overall, mode = 'prob')
    sc_contacts, xyz = load_sc_contacts(dir, N_max = None, triu = True,
                                    gaussian = True, zero_diag = True, jobs = 8,
                                    down_sampling = 3, sparsify = False,
                                    return_xyz = True)
    print_size(xyz, 'xyz')
    print_size(sc_contacts, 'sc_contacts')
    N, _ = sc_contacts.shape
    for i in range(1):
        # diag processing
        t0 = time.time()
        sc_contacts_diag = diagonal_preprocessing_bulk(sc_contacts, mean_per_diag,
                                                        triu = True)
        tf = time.time()
        print_time(t0, tf, 'diag')

        # compute distance
        t0 = time.time()
        dist = dmaps.DistanceMatrix(sc_contacts_diag)
        dist.compute(metric=dmaps.metrics.correlation)
        D = dist.get_distances()
        plot_matrix(D, ofile = osp.join(odir, 'distances.png'),
                        vmin = 'min', vmax = 'max')
        tf = time.time()
        print_time(t0, tf, 'distance')

        # compute eigenvectors
        dmap = dmaps.DiffusionMap(dist)
        eps = tune_epsilon(dmap, osp.join(odir, 'tuning.png'))
        dmap.set_kernel_bandwidth(eps)
        dmap.compute(5, 0.5)

        v = dmap.get_eigenvectors()
        w = dmap.get_eigenvalues()
        print('w', w)
        order = np.argsort(v[:, 1])
        print('\n\norder corr: ', pearson_round(order, np.arange(0, N, 1), stat = 'spearman'))

        plot_eigenvectors(v, xyz.reshape(-1, m, 3), sc_contacts, odir, i)


def plot_gif_michrom():
    dir = '/home/erschultz/michrom/project/chr_05/chr_05_02/'
    filenames = [osp.join(dir, f'cluster{i}_contacts.png') for i in [0, 3, 4, 5, 2, 1]]
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))


    imageio.mimsave(osp.join(dir, 'clusters.gif'), frames, format='GIF', fps=1)

    # # remove files
    # for filename in set(filenames):
    #     os.remove(filename)


if __name__ == '__main__':
    main_sc()
    # plot_gif_michrom()
