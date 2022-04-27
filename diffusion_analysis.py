import argparse
import multiprocessing
import os
import os.path as osp
import sys
import time
from shutil import move, rmtree

import imageio
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from utils.argparse_utils import str2bool, str2None
from utils.load_utils import save_sc_contacts
from utils.plotting_utils import plot_matrix, plot_sc_contact_maps_inner
from utils.utils import (DiagonalPreprocessing, pearson_round, print_size,
                         print_time, triu_to_full)
from utils.xyz_utils import (find_dist_between_centroids, find_label_centroid,
                             lammps_load, xyz_load, xyz_to_contact_grid)

import dmaps  # https://github.com/ERSchultz/dmaps


def getArgs(default_dir='/home/erschultz/dataset_test/samples/sample92'):
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--dir', type=str, default=default_dir,
                        help='location of data')
    parser.add_argument('--scratch', type=str2None, default='/home/erschultz/scratch',
                        help='scratch dir')
    parser.add_argument('--odir', type=str,
                        help='location to write to')
    parser.add_argument('--N_min', type=int, default=2000,
                        help='minimum sample index to keep')
    parser.add_argument('--mode', type=str, default='contact_diffusion')
    parser.add_argument('--update_mode', type=str, default='eig')
    parser.add_argument('--jobs', type=int, default=15)
    parser.add_argument('--sparse_format', action='store_true',
                        help='True to store sc_contacts in sparse format')
    parser.add_argument('--down_sampling', type=int, default=5)
    parser.add_argument('--its', type=int, default=1,
                        help='number of iterations')
    parser.add_argument('--chunk_size', type=int, default=300,
                        help='chunk size for pairwise_distances_chunk')
    parser.add_argument('--plot', type=str2bool, default=False,
                        help='True to plot')

    args = parser.parse_args()
    if args.odir is None:
        args.odir = args.dir
    elif not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)

    odir = osp.join(args.odir, args.mode)
    if osp.exists(odir):
        rmtree(odir)

    args.scratch_dir = osp.join(args.scratch, args.mode)
    if osp.exists(args.scratch_dir):
        rmtree(args.scratch_dir)
    os.mkdir(args.scratch_dir, mode = 0o755)

    args.log_file_path = osp.join(args.scratch_dir, 'out.log')
    args.log_file = open(args.log_file_path, 'a')

    print(args, file = args.log_file)
    return args

class Updater():
    def __init__(self, k, log_file = sys.stdout):
        self.k = k
        self.log_file = log_file

    def update_eig_chunk(self, vi, dir, odir):
        '''
        Update sc contact maps based on eigenvector vi.
        Average contacts with range k.

        This version only loads a chunk of sc_contacts of size (2l+1) into RAM at a time.
        '''
        t0 = time.time()
        where = np.argsort(vi)
        N = len(vi)
        odir = osp.join(odir, 'sc_contacts')
        if not osp.exists(odir):
            os.mkdir(odir, mode = 0o755)

        # merge adjacent contacts
        y_queue = [] # queue of rotating (2l+1) contacts
        # initialize queue
        for j in range(self.k):
            fpath = osp.join(dir, f'y_sc_{where[j]}.npy')
            y_queue.append(np.load(fpath))
        # iterate through
        for i in range(N):
            if i < N-self.k:
                fpath = osp.join(dir, f'y_sc_{where[i+self.k]}.npy')
                y_queue.append(np.load(fpath))

            # calculate average
            y_new = np.mean(y_queue, axis = 0)
            np.save(osp.join(odir, f'y_sc_{where[i]}.npy'), y_new)

            # pop first entry of queue
            if i >= self.k:
                y_queue.pop(0)


        tf = time.time()
        print_time(t0, tf, 'update', file = self.log_file)

    def update_eig(self, vi, sc_contacts):
        '''
        Update sc contact maps based on eigenvector vi.
        Average contacts with range k.
        '''
        t0 = time.time()
        where = np.argsort(vi)
        N, _ = sc_contacts.shape

        # sort sc_contacts
        sc_contacts = sc_contacts[where]

        # merge adjacent contacts
        if self.k == 1:
            # modify in place to preserve RAM
            # I did test this - it works
            for i in range(N):
                new = np.mean(sc_contacts[max(0, i-1):min(i+self.k+1, N), :], axis = 0)
                # print(new)
                if i > 0:
                    sc_contacts[i-1, :] = new_prev
                new_prev = new.copy()
            sc_contacts[N-1] = new_prev
        else:
            sc_contacts_new = np.zeros_like(sc_contacts)
            for i in range(N):
                new = np.mean(sc_contacts[max(0, i-1):min(i+self.k+1, N), :], axis = 0)
                sc_contacts_new[i] = new

            sc_contacts = sc_contact_new

        # undo sort
        order = np.argsort(where)
        sc_contacts = sc_contacts[order]
        tf = time.time()
        print_time(t0, tf, 'update', file = self.log_file)

        return sc_contacts

    def update_kNN(self, v, sc_contacts):
        raise NotImplementedError

def tune_epsilon(input, ofile):
    X = np.arange(-12, 20, 1)
    epsilons = np.exp(X)
    Y = np.zeros_like(epsilons)
    for i, eps in enumerate(epsilons):
        if isinstance(input, dmaps.DiffusionMap):
            Y[i] = np.log(input.sum_similarity_matrix(eps))
        elif isinstance(input, np.ndarray):
            Y[i] = np.sum(np.exp(-1/2 * input**2 / eps))
        else:
            print(type(input))

    # find best linear regression on subset of 4/5 points
    best_indices = (-1, -1)
    best_score = 0
    best_reg = None
    for size in [4, 5]:
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
            elif best_reg is None:
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

def plot_eigenvectors(v, xyz, odir):
    N = len(v)

    # plot first 2 nonzero eigenvectors, color by order
    sc = plt.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_2$', fontsize = 16)
    plt.ylabel(r'$v_3$', fontsize = 16)
    plt.savefig(osp.join(odir, 'projection23.png'.strip('_')))
    plt.close()

    # plot 3 and 4 eigenvectors, color by order
    sc = plt.scatter(v[:,2]/v[:,0], v[:,3]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.savefig(osp.join(odir, 'projection34.png'.strip('_')))
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
    plt.savefig(osp.join(odir, 'projection234.png'.strip('_')))
    plt.close()

    # k_means
    num_vecs = 3
    X = v[:,1:1+num_vecs]
    sil_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k).fit(X)
        sil_scores.append(silhouette_score(X, kmeans.labels_))
    plt.plot(np.arange(2, 10, 1), sil_scores)
    plt.savefig(osp.join(odir, 'k_means_sil_score.png'.strip('_')))
    plt.close()
    k = np.argmax(sil_scores) + 2
    print(f'Using k = {k}')
    kmeans = KMeans(n_clusters=k).fit(X)

    # plot average contact map within cluster
    for cluster in range(k):
        ind = np.argwhere(kmeans.labels_ == cluster)
        xyz_ind = xyz[ind].reshape(len(ind), -1, 3)
        y_cluster = xyz_to_contact_grid(xyz_ind, 28.7)
        plot_matrix(y_cluster, osp.join(odir, f'cluster{cluster}_contacts.png'),
                    vmax = 'mean', title = f'cluster {cluster}')

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
    plt.savefig(osp.join(odir, 'projection23_kmeans.png'.strip('_')))
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
    plt.savefig(osp.join(odir, 'projection34_kmeans.png'.strip('_')))
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
    plt.savefig(osp.join(odir, 'projection234_kmeans.png'.strip('_')))
    plt.close()

def plot_contacts(dir, order, args):
    for order, folder in zip([range(args.N), order], ['sc_contacts_time', 'sc_contacts_traj']):
        if order is None:
            continue
        odir = osp.join(dir, folder)
        if not osp.exists(odir):
            os.mkdir(odir, mode = 0o775)

        # make plots
        filenames = []
        mapping = []
        for i, file_i in enumerate(order):
            if i % (args.N // 20) == 0:
                y = np.load(osp.join(dir, f'y_sc_{file_i}.npy'))
                y = triu_to_full(y, args.m)
                ofile = osp.join(odir, f'y_sc_{i}.png')
                filenames.append(ofile)
                if args.jobs > 1:
                    mapping.append((y, ofile, i, 0, 'max'))
                else:
                    plot_matrix(y, ofile, title = i, vmax = 'max')

        if args.jobs > 1:
            with multiprocessing.Pool(args.jobs) as p:
                p.starmap(plot_matrix, mapping)

        # make gif
        print('starting gif')
        frames = []
        for filename in filenames:
            frames.append(imageio.imread(filename))

        imageio.mimsave(osp.join(odir, 'sc_contacts.gif'), frames, format='GIF', fps=2)

def load_helper(args, contacts = False):
    xyz_file = osp.join(args.dir, 'data_out/output.xyz')
    lammps_file = osp.join(args.dir, 'traj.dump.lammpstrj')
    if osp.exists(xyz_file):
        xyz = xyz_load(xyz_file,
                    multiple_timesteps = True, save = True, N_min = 10,
                    down_sampling = args.down_sampling)
    elif osp.exists(lammps_file):
        xyz = lammps_load(lammps_file, save = False, N_min = args.N_min,
                        down_sampling = args.down_sampling)

    if contacts:
        args.sc_contacts_dir = osp.join(args.odir, 'sc_contacts')
        save_sc_contacts(xyz, args.sc_contacts_dir, args.jobs, sparsify = True,
                        overwrite = True)

    return xyz

def diag_processsing_chunk(dir, odir, args):
    t0 = time.time()
    if odir is not None and not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    overall = np.zeros(int(args.m*(args.m+1)/2))
    for i in range(args.N):
        fpath = osp.join(dir, f'y_sc_{i}.npy')
        overall += np.load(fpath)
    overall = triu_to_full(overall)
    mean_per_diag = DiagonalPreprocessing.genomic_distance_statistics(overall, mode = 'prob')
    print(mean_per_diag, file = args.log_file)
    sc_contacts_diag = DiagonalPreprocessing.process_chunk(dir, mean_per_diag, odir,
                                    jobs = 1, sparse_format = args.sparse_format)

    tf = time.time()
    print_time(t0, tf, 'diag', file = args.log_file)

    return sc_contacts_diag

def diag_processsing(dir, odir, args):
    t0 = time.time()

    sc_contacts = np.zeros((args.N, int(args.m*(args.m+1)/2)))
    for i in range(args.N):
        fpath = osp.join(dir, f'y_sc_{i}.npy')
        sc_contacts[i] = np.load(fpath)
    overall = np.sum(sc_contacts, axis = 0)
    overall = triu_to_full(overall)
    mean_per_diag = DiagonalPreprocessing.genomic_distance_statistics(overall, mode = 'prob')
    sc_contacts_diag = DiagonalPreprocessing.process_bulk(sc_contacts, mean_per_diag,
                                                    triu = True)
    tf = time.time()
    print_time(t0, tf, 'diag')

    return sc_contacts_diag

def pairwise_distances_chunk(dir, m, metric, chunk_size):
    N = len([f for f in os.listdir(dir) if f.endswith('.npy')])
    files = [f'y_sc_{i}.npy' for i in range(N)]
    D = np.zeros((N, N))
    for i in range(0, N, chunk_size):
        i_len = len(files[i:i + chunk_size])

        X = np.zeros((i_len, int(m*(m+1)/2)))
        for k, file in enumerate(files[i:i + chunk_size]):
            X[k] = np.load(osp.join(dir, file))

        for j in range(i, N, chunk_size):
            j_len = len(files[j:j + chunk_size])
            if j == i:
                Y = X
            else:
                Y = np.zeros((j_len, int(m*(m+1)/2)))
                for k, file in enumerate(files[j:j + chunk_size]):
                    Y[k] = np.load(osp.join(dir, file))

            D_ij = pairwise_distances(X, Y, metric = metric)
            D[i:i+i_len, j:j+j_len] = D_ij

    # make symmetric
    D = np.triu(D) + np.triu(D, 1).T

    return D

class GaussianProcessing():
    def __init__(self, args):
        self.idir = args.sc_contacts_dir
        self.odir = osp.join(args.odir_i, 'sc_contacts')
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)

        self.m = args.m

    def process(self, jobs, log_file = sys.stdout):
        t0 = time.time()

        mapping = []
        for file in os.listdir(self.idir):
            ifile = osp.join(self.idir, file)
            ofile = osp.join(self.odir, file)
            mapping.append((ifile, ofile))

        if jobs > 1:
            with multiprocessing.Pool(jobs) as p:
                p.starmap(self.inner, mapping)
        else:
            for ifile, ofile, m in mapping:
                self.inner(ifile, ofile)

        tf = time.time()
        print_time(t0, tf, 'gaussian', file = log_file)

    def inner(self, ifile, ofile):
        y = np.load(ifile)
        y = triu_to_full(y, self.m)
        y = gaussian_filter(y, sigma = 4)
        y = y[np.triu_indices(self.m)]
        np.save(ofile, y)

def xyz_diffusion():
    args = getArgs()
    xyz = load_helper(args)

    # load xyz
    t0 = time.time()


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

def contact_diffusion():
    args = getArgs()
    xyz = load_helper(args, True)
    args.N, args.m, _ = xyz.shape

    order = None
    odir_prev = None
    updater = Updater(1, args.log_file)
    for it in range(args.its+1):
        args.odir_i = osp.join(args.scratch_dir, f'iteration_{it}')
        os.mkdir(args.odir_i, mode = 0o755)
        print(f"Iteration {it}")
        print(f"Iteration {it}", file = args.log_file)
        if it > 0:
            # diag processing
            diag_dir = osp.join(args.odir_i, 'sc_contacts_diag')
            sc_contacts_diag = diag_processsing_chunk(osp.join(odir_prev, 'sc_contacts'),
                            diag_dir, args)
            print_size(sc_contacts_diag, 'sc_contacts_diag', file = args.log_file)

            # compute distance
            t0 = time.time()
            D = pairwise_distances_chunk(diag_dir, args.m, metric = 'cosine',
                                        chunk_size = args.chunk_size)
            # D = (sc_contacts_diag, sc_contacts_diag,
            #                         metric = 'cosine')
            del sc_contacts_diag # no longer needed
            plot_matrix(D, ofile = osp.join(args.odir_i, 'distances.png'),
                            vmin = 'min', vmax = 'mean')
            tf = time.time()
            print_time(t0, tf, 'distance', file = args.log_file)

            # compute eigenvectors
            dmap = dmaps.DiffusionMap(D)
            eps = tune_epsilon(dmap, osp.join(args.odir_i, 'tuning.png'))
            dmap.set_kernel_bandwidth(eps)
            dmap.compute(5, 0.5)

            v = dmap.get_eigenvectors()
            w = dmap.get_eigenvalues()
            print('w', w, file = args.log_file)
            order = np.argsort(v[:, 1])
            print('\norder corr: ',
                pearson_round(order, np.arange(0, args.N, 1), stat = 'spearman'),
                file = args.log_file)

            np.savetxt(osp.join(args.odir_i, 'order.txt'), order, fmt='%i')
            if args.plot:
                plot_eigenvectors(v, xyz, args.odir_i)

            if args.update_mode == 'eig':
                sc_contacts = updater.update_eig_chunk(v[:, 1],
                                        osp.join(odir_prev, 'sc_contacts'),
                                        args.odir_i)
            elif args.update_mode == 'knn':
                sc_contacts = updater.update_kNN(v, sc_contacts)
        else:
            # apply gaussian
            GP = GaussianProcessing(args)
            GP.process(args.jobs, args.log_file)

        # Plots
        if args.plot:
            t0 = time.time()
            plot_contacts(osp.join(args.odir_i, 'sc_contacts'), order, args)
            tf = time.time()
            print_time(t0, tf, 'plot', file = args.log_file)
        print('\n', file = args.log_file)
        odir_prev = args.odir_i

    # move files from scratch to odir
    move(args.scratch_dir, args.odir)

def contact_laplacian():
    args = getArgs()
    xyz = load_helper(args, True)
    args.N, args.m, _ = xyz.shape

    order = None
    odir_prev = None
    for it in range(args.its+1):
        args.odir_i = osp.join(args.scratch_dir, f'iteration_{it}')
        os.mkdir(args.odir_i, mode = 0o755)
        print(f"Iteration {it}")
        if it > 0:
            # diag processing
            sc_contacts_diag = diag_processsing(osp.join(odir_prev, 'sc_contacts'),
                            osp.join(args.odir_i, 'sc_contacts_diag'), args)
            print_size(sc_contacts_diag, 'sc_contacts_diag')

            # compute distance
            t0 = time.time()
            D = pairwise_distances(sc_contacts_diag, sc_contacts_diag,
                                            metric = 'correlation')
            del sc_contacts_diag # no longer needed
            plot_matrix(D, ofile = osp.join(args.odir_i, 'distances.png'),
                            vmin = 'min', vmax = 'max')
            tf = time.time()
            print_time(t0, tf, 'distance')

            # compute adjacency
            eps_final = tune_epsilon(D, ofile = osp.join(args.odir_i, 'tuning.png'))
            A = np.exp(-1/2 * D**2 / eps_final)
            plot_matrix(A, ofile = osp.join(args.odir_i, 'A.png'),
                            vmin = 'min', vmax = 'max')

            # compute laplacian
            A_tilde = laplacian(A, normed = True)
            plot_matrix(A_tilde, ofile = osp.join(args.odir_i, 'A_tilde.png'),
                            vmin = np.min(A_tilde), vmax = np.max(A_tilde),
                            cmap = 'blue-red')
            np.savetxt(osp.join(args.odir_i, 'A_tilde.txt'), A_tilde, fmt='%.2f')

            # compute eigenvectors
            w, v = np.linalg.eig(A_tilde)
            v = v[:, np.argsort(w)]
            w = np.sort(w)
            print('w', w[:10])
            # get first nonzero eigenvector of A_tilde
            lmbda = w[0]; i = 0
            while lmbda <= 1e-12 and i < len(w):
                i += 1
                lmbda = w[i]
            assert i == 1, "first nonzero eigenvector is not 2nd vector"
            order = np.argsort(v[:, i])
            print('\n\norder corr: ', pearson_round(order, np.arange(0, N, 1),
                                                    stat = 'spearman'))

            np.savetxt(osp.join(args.odir_i, 'order.txt'), order, fmt='%i')
            plot_eigenvectors(v, xyz, args.odir_i)

            if args.update_mode == 'eig':
                sc_contacts = Updater.update_eig_chunk(v[:, 1], osp.join(odir_prev, 'sc_contacts'), args.odir_i)
        else:
            # apply gaussian
            GP = GaussianProcessing(args)
            GP.process(args.jobs)

        # Plots
        t0 = time.time()
        plot_contacts(osp.join(args.odir_i, 'sc_contacts'), order, args)
        tf = time.time()
        print_time(t0, tf, 'plot')
        print('\n')
        odir_prev = args.odir_i

    # move files from scratch to odir
    move(args.scratch_dir, args.odir)

def plot_gif_michrom():
    dir = '/home/erschultz/michrom/project/chr_05/chr_05_02_copy/contact_diffusion'
    filenames = [osp.join(dir, f'cluster{i}_contacts.png') for i in [1, 3, 0, 4, 2]]
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))


    imageio.mimsave(osp.join(dir, 'clusters.gif'), frames, format='GIF', fps=1)

    # # remove files
    # for filename in set(filenames):
    #     os.remove(filename)

def main():
    args = getArgs()
    if args.mode == 'xyz_diffusion':
        xyz_diffusion()
    elif args.mode == 'contact_diffusion':
        contact_diffusion()
    elif args.mode == 'contact_laplacian':
        contact_laplacian()


if __name__ == '__main__':
    main()
    # plot_gif_michrom()
