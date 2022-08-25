import argparse
import json
import multiprocessing
import os
import os.path as osp
import sys
import time
from shutil import copyfile, move, rmtree

import hicrep
import hicrep.utils
import imageio.v2 as imageio
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from utils.argparse_utils import ArgparserConverter
from utils.load_utils import save_sc_contacts
from utils.plotting_utils import plot_matrix
from utils.utils import (SCC, DiagonalPreprocessing, InnerProduct,
                         pearson_round, print_size, print_time, triu_to_full)
from utils.xyz_utils import lammps_load, xyz_load, xyz_to_contact_grid

import dmaps  # https://github.com/ERSchultz/dmaps


def getArgs(default_dir='/home/erschultz/dataset_test_sc_traj/samples/combined'):
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    # directories
    parser.add_argument('--dir', type=str, default=default_dir,
                        help='location of data')
    parser.add_argument('--scratch', type=AC.str2None, default='/home/erschultz/scratch',
                        help='scratch dir')
    parser.add_argument('--odir', type=str,
                        help='location to write to')

    # data arguments
    parser.add_argument('--experimental', type=AC.str2bool, default=False,
                        help='True if using experimental data')
    parser.add_argument('--N_min', type=int, default=2000,
                        help='minimum sample index to keep')
    parser.add_argument('--input_file_type', type = str, default='npy',
                        help='file format in {mcool, npy}')
    parser.add_argument('--down_sampling', type=int, default=50)
    # required iff input_file_type == .mcool
    parser.add_argument('--resolution', type=int, default=50000,
                        help='resoultion for mcool file')
    parser.add_argument('--chrom', type=str, default='10',
                        help='specify chromosome for mcool file')

    # algorithm arguments
    parser.add_argument('--mode', type=str, default='contact_diffusion')
    parser.add_argument('--update_mode', type=str, default='kNN')
    parser.add_argument('--preprocessing_mode', type=str, default='identity')
    parser.add_argument('--k', type=int, default=2,
                        help='k for update_mode')
    parser.add_argument('--jobs', type=int, default=15)
    parser.add_argument('--sparse_format', action='store_true',
                        help='True to store sc_contacts in sparse format')
    parser.add_argument('--its', type=int, default=1,
                        help='number of iterations')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='chunk size for pairwise_distances_chunk')
    parser.add_argument('--plot', type=AC.str2bool, default=True,
                        help='True to plot');
    parser.add_argument('--metric', type=str, default='cosine',
                        help='distance metric to use')

    args = parser.parse_args()
    if args.odir is None:
        args.odir = args.dir
    elif not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)

    fname = f'{args.mode}_{args.update_mode}{args.k}{args.metric}'
    args.odir_mode = osp.join(args.odir, fname)
    if osp.exists(args.odir_mode):
        rmtree(args.odir_mode)

    args.scratch_dir = osp.join(args.scratch, fname)
    if osp.exists(args.scratch_dir):
        rmtree(args.scratch_dir)
    os.mkdir(args.scratch_dir, mode = 0o755)

    args.log_file_path = osp.join(args.scratch_dir, 'out.log')
    args.log_file = open(args.log_file_path, 'a')

    # ensure consistent formatting
    args.update_mode = args.update_mode.lower()
    args.input_file_type = args.input_file_type.strip('.')

    if args.metric == 'scc':
        assert args.preprocessing_mode != 'gaussian', "don't use gaussian with SCC"

    print(args, file = args.log_file)
    print(args)
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

    def update_kNN_chunk(self, v, dir, odir):
        '''
        Update sc contact maps based on kNN in eigenspace v.

        Only loads a chunk of sc_contacts of size (k+1) into RAM at a time.
        '''
        t0 = time.time()
        N = len(v)
        odir = osp.join(odir, 'sc_contacts')
        if not osp.exists(odir):
            os.mkdir(odir, mode = 0o755)

        # compute pairwise distance in v space
        D = euclidean_distances(v, v)

        # merge adjacent contacts
        for i in range(N):
            y_list = []
            nn = np.argpartition(D[i], self.k+1)[:self.k+1] # include self + kNN
            for j in nn:
                y_list.append(np.load(osp.join(dir, f'y_sc_{j}.npy')))

            # calculate average
            y_new = np.mean(y_list, axis = 0)
            np.save(osp.join(odir, f'y_sc_{i}.npy'), y_new)

        tf = time.time()
        print_time(t0, tf, 'update', file = self.log_file)

## plotting functions
def plot_eigenvectors_inner(v, odir, fig_label, labels = None):
    N = len(v)
    if labels is not None:
        if np.min(labels) == 1:
            labels -= 1
            # switch to zero based indexing
        k = len(np.unique(labels))
        if k <= 10:
            cmap = matplotlib.cm.get_cmap('tab10')
        else:
            cmap = matplotlib.cm.get_cmap('tab20')
        ind = np.arange(k) % cmap.N
        colors = plt.cycler('color', cmap(ind))

    # eig 2 and 3
    if labels is None:
        sc = plt.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], c = np.arange(0, N, 1))
        plt.colorbar(sc)
    else:
        for cluster, c in zip(range(k), colors):
            ind = np.argwhere(labels == cluster)
            plt.scatter(v[ind, 1]/v[ind, 0], v[ind, 2]/v[ind, 0], color = c['color'],
                        label = cluster)
        plt.legend()
    plt.xlabel(r'$v_2$', fontsize = 16)
    plt.ylabel(r'$v_3$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, f'projection23_{fig_label}.png'))
    plt.close()

    # eig 3 and 4
    if labels is None:
        sc = plt.scatter(v[:,2]/v[:,0], v[:,3]/v[:,0], c = np.arange(0, N, 1))
        plt.colorbar(sc)
    else:
        for cluster, c in zip(range(k), colors):
            ind = np.argwhere(labels == cluster)
            plt.scatter(v[ind, 2]/v[ind, 0], v[ind, 3]/v[ind, 0], color = c['color'],
                        label = cluster)
        plt.legend()
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, f'projection34_{fig_label}.png'))
    plt.close()

    # eig 2, 3, 4
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if labels is None:
        sc = ax.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], v[:,3]/v[:,0],
                        c = np.arange(0, N, 1))
        plt.colorbar(sc)
    else:
        for cluster, c in zip(range(k), colors):
            ind = np.argwhere(labels == cluster)
            ax.scatter(v[ind,1]/v[ind,0], v[ind,2]/v[ind,0], v[ind,3]/v[ind,0],
                        color = c['color'], label = cluster)
        plt.legend()
    ax.set_xlabel(r'$v_2$', fontsize = 16)
    ax.set_ylabel(r'$v_3$', fontsize = 16)
    ax.set_zlabel(r'$v_4$', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, f'projection234_{fig_label}.png'))
    plt.close()

def plot_eigenvectors(v, xyz, odir):
    N = len(v)

    # color by order
    plot_eigenvectors_inner(v, odir, 'time')

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
    print(f'Using k = {k} for k_means')
    kmeans = KMeans(n_clusters=k).fit(X)

    # color by kmeans
    plot_eigenvectors_inner(v, odir, 'kmeans', kmeans.labels_)

    if xyz is not None:
        _, _, d = xyz.shape

        # plot average contact map within cluster
        # TODO this requires xyz, but shouldn't
        for cluster in range(k):
            ind = np.argwhere(kmeans.labels_ == cluster)
            xyz_ind = xyz[ind].reshape(len(ind), -1, d)
            y_cluster = xyz_to_contact_grid(xyz_ind, 28.7)
            plot_matrix(y_cluster, osp.join(odir, f'cluster{cluster}_contacts.png'),
                        vmax = 'mean', title = f'cluster {cluster}')

        # color by ground truth label
        if d > 3: # ground truth exists
            plot_eigenvectors_inner(v, odir, 'ground_truth', xyz[:, 0, 3])

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
            if i % (args.N // 10) == 0:
                y = None
                if args.input_file_type == 'mcool':
                    ifile = osp.join(dir, f'y_sc_{file_i}.mcool')
                    if osp.exists(ifile):
                        c, binsize = hicrep.utils.readMcool(ifile, args.resolution)
                        y = c.matrix(balance=False).fetch(f'{args.chrom}')
                elif args.input_file_type == 'npy':
                    ifile = osp.join(dir, f'y_sc_{file_i}.npy')
                    if osp.exists(ifile):
                        y = np.load(ifile)
                        if len(y.shape) == 1:
                            y = triu_to_full(y, args.m)

                if y is None:
                    continue

                contacts = int(np.sum(y) / 2)
                sparsity = np.round(np.count_nonzero(y) / len(y)**2 * 100, 2)
                title = f'Sample {i}:\n# contacts: {contacts}, sparsity: {sparsity}%'

                ofile = osp.join(odir, f'y_sc_{i}.png')
                filenames.append(ofile)
                if args.jobs > 1:
                    mapping.append((y, ofile, title, 0, 'abs_max'))
                else:
                    plot_matrix(y, ofile, title, vmax = 'abs_max')

        if args.jobs > 1:
            with multiprocessing.Pool(args.jobs) as p:
                p.starmap(plot_matrix, mapping)

        # make gif
        print('starting gif')
        frames = []
        for filename in filenames:
            frames.append(imageio.imread(filename))

        imageio.mimsave(osp.join(odir, 'sc_contacts.gif'), frames, format='GIF', fps=2)
## end section

def diag_processsing_chunk(dir, odir, args):
    if odir is not None and not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    # use overall contact map to get mean_per_diag
    t0 = time.time()
    overall = np.zeros(int(args.m*(args.m+1)/2))
    for i in range(args.N):
        fpath = osp.join(dir, f'y_sc_{i}.npy')
        overall += np.load(fpath)
        if np.isnan(overall[10]):
            print('here', i, overall)
            print(np.load(fpath))
            break
    overall = triu_to_full(overall)

    mean_per_diag = DiagonalPreprocessing.genomic_distance_statistics(overall, mode = 'prob')
    print('mean_per_diag:', mean_per_diag, file = args.log_file)

    tf = time.time()
    print_time(t0, tf, 'mean_per_diag', file = args.log_file)

    # process diag
    t0 = time.time()
    DiagonalPreprocessing.process_chunk(dir, mean_per_diag, odir,
                                chunk_size = args.chunk_size, jobs = args.jobs,
                                sparse_format = args.sparse_format)

    tf = time.time()
    print_time(t0, tf, 'diag', file = args.log_file)

class PreProcessing():
    def __init__(self, args, files):
        self.mode = args.preprocessing_mode
        self.resolution = args.resolution
        self.chrom = args.chrom
        self.file_type = args.input_file_type
        self.files = files
        self.odir = osp.join(args.odir_i, 'sc_contacts')
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)

        self.m = args.m

    def process(self, jobs, log_file = sys.stdout):
        t0 = time.time()

        mapping = []
        ifile_dict = {}
        i = 0
        for ifile in self.files:
            if not ifile.endswith('.' + self.file_type):
                continue
            ofile = osp.join(self.odir, f'y_sc_{i}.{self.file_type}')
            ifile_dict[ifile] = i
            mapping.append((ifile, ofile))
            i += 1

        # save ifile_dict
        with open(osp.join(self.odir, 'ifile_dict.json'), 'w') as f:
            json.dump(ifile_dict, f, indent = 2)

        if self.mode == 'gaussian':
            fn = self.gaussian
        elif self.mode == 'identity':
            fn = self.identity
        elif self.mode == 'sparsity_filter':
            fn = self.sparsity_filter
        elif self.mode == 'triu':
            fn = self.to_triu
        else:
            raise Exception(f"Unrecognized mode: {self.mode}")

        if jobs > 1:
            with multiprocessing.Pool(jobs) as p:
                p.starmap(fn, mapping)
        else:
            for ifile, ofile in mapping:
                fn(ifile, ofile)

        tf = time.time()
        print_time(t0, tf, 'preprocessing', file = log_file)

    def gaussian(self, ifile, ofile):
        if self.file_type == 'npy':
            y = np.load(ifile)
        else:
            raise Exception(f'Unaccepted file type: {self.file_type}')
        if len(y.shape) == 1:
            y = triu_to_full(y, self.m)
        y_gauss = gaussian_filter(y, sigma = 4)
        y_gauss = y_gauss[np.triu_indices(self.m)]
        np.save(ofile, y_gauss)

    def identity(self, ifile, ofile):
        # TODO avoid doing this by just using the path to the ifile
        copyfile(ifile, ofile)

    def to_triu(self, ifile, ofile):
        y = self.load_file(ifile)
        y = y[np.triu_indices(self.m)]
        np.save(ofile, y)

    def sparsity_filter(self, ifile, ofile):
        y = self.load_file(ifile)

        if len(y.shape) == 1:
            sparsity = np.count_nonzero(y) / len(y) * 100
            # TODO test this against dense version
        else:
            sparsity = np.count_nonzero(y) / len(y)**2 * 100

        if sparsity > 0.1:
            y = y[np.triu_indices(self.m)]
            np.save(ofile, y)

    def load_file(self, ifile):
        if self.file_type == 'mcool':
            c, _ = hicrep.utils.readMcool(ifile, self.resolution)
            y = c.matrix(balance=False).fetch(f'{self.chrom}')
        elif self.file_type == 'npy':
            y = np.load(ifile)
        else:
            raise Exception(f'Unaccepted file type: {self.file_type}')

        return y


class Diffusion():
    def __init__(self, args, contacts = False):
        self.args = args
        self.files = [] # keep track of current list files to use
        # load data
        t0 = time.time()
        if args.experimental:
            self.xyz = None
            assert contacts
            files = sorted(os.listdir(args.dir), key = lambda x: int(x.split('.')[1]))

            for file in files[::args.down_sampling]:
                if args.input_file_type == 'mcool':
                    file_path = osp.join(args.dir, file, 'adj.mcool')
                elif args.input_file_type == 'npy':
                    file_path = osp.join(args.dir, file, 'y.npy')
                if osp.exists(file_path):
                    self.files.append(file_path)

            args.N = len(self.files)
            if args.input_file_type == 'npy':
                y = np.load(self.files[-1])
                args.m = len(y)
            else:
                args.m = None

            # get N
            print(f'N={args.N}', file = args.log_file)
        else:
            npy_file = osp.join(args.dir, 'xyz.npy')
            xyz_file = osp.join(args.dir, 'data_out/output.xyz')
            lammps_file = osp.join(args.dir, 'traj.dump.lammpstrj')
            if osp.exists(npy_file):
                self.xyz = np.load(npy_file)
                self.xyz = self.xyz[::args.down_sampling]
            elif osp.exists(xyz_file):
                self.xyz = xyz_load(xyz_file,
                            multiple_timesteps = True, save = True, N_min = 10,
                            down_sampling = args.down_sampling)
            elif osp.exists(lammps_file):
                self.xyz = lammps_load(lammps_file, save = False, N_min = args.N_min,
                                down_sampling = args.down_sampling)

            args.N, args.m, _ = self.xyz.shape
            print(f'N={args.N}, m = {args.m}', file = args.log_file)

            if contacts:
                args.sc_contacts_dir = osp.join(args.odir, 'sc_contacts')
                save_sc_contacts(self.xyz, args.sc_contacts_dir, args.jobs, sparsify = True,
                                overwrite = True)

        tf = time.time()
        print_time(t0, tf, 'load')

    def compute_distance(self, input, metric):
        args = self.args
        t0 = time.time()
        try:
            if isinstance(metric, str):
                if metric == 'scc':
                    D = self.pairwise_distances_parallel(input, metric)
                elif metric == 'inner_product':
                    IP = InnerProduct(input, 10, jobs = args.jobs)
                    D = IP.get_distance_matrix()
                else:
                    D = self.pairwise_distances_chunk(input, metric)
            else:
                dist = dmaps.DistanceMatrix(input)
                dist.compute(metric=metric)
                D = dist.get_distances()
        except Exception as e:
            print(e, file = args.log_file)
            raise
            return None

        plot_matrix(D, ofile = osp.join(args.odir_i, 'distances.png'),
                        vmin = 'min', vmax = 'max')
        np.savetxt(osp.join(args.odir_i, 'distances.txt'), D)
        tf = time.time()
        print_time(t0, tf, 'distance', file = args.log_file)

        return D

    def pairwise_distances_parallel(self, input_dir, metric):
        scc = SCC()
        args = self.args
        files = [f for f in os.listdir(input_dir) if f.endswith(args.input_file_type)]
        N = len(files)
        D = np.zeros((N, N))
        mapping = []
        for i in range(0, N):
            ifile = osp.join(input_dir, files[i])
            for j in range(i, N):
                # note that this ordering of the double loop is essential for triu_to_full to work
                jfile = osp.join(input_dir, files[j])
                mapping.append((ifile, jfile, 1, 10, True))

        assert metric == 'scc', f'other metrics not supported yet: {metric}'
        with multiprocessing.Pool(args.jobs) as p:
            result = np.array(p.starmap(scc.scc_file, mapping))

        # make symmetric and convert to distance
        D = 1 - triu_to_full(result)

        return D

    def pairwise_distances_chunk(self, input_dir, metric):
        scc = SCC()
        args = self.args
        files = [f for f in os.listdir(input_dir) if f.endswith(args.input_file_type)]
        print(files)
        N = len(files)
        D = np.zeros((N, N))
        for i in range(0, N, args.chunk_size):
            i_len = len(files[i:i + args.chunk_size])

            # get X chunk
            X = np.zeros((i_len, int(args.m*(args.m+1)/2)))
            for k, file in enumerate(files[i:i + args.chunk_size]):
                X[k] = np.load(osp.join(input_dir, file))

            # get Y chunk
            for j in range(i, N, args.chunk_size):
                j_len = len(files[j:j + args.chunk_size])
                if j == i:
                    Y = X
                else:
                    Y = np.zeros((j_len, int(args.m*(args.m+1)/2)))
                    for k, file in enumerate(files[j:j + args.chunk_size]):
                        Y[k] = np.load(osp.join(input_dir, file))

                if metric == 'scc':
                    D_ij = pairwise_distances(X, Y, metric = scc.scc, h = 1, K = 10,
                                                var_stabilized = True,
                                                n_jobs = args.jobs)
                    D_ij = 1 - D_ij # convert to distance
                else:
                    D_ij = pairwise_distances(X, Y, metric = metric)
                D[i:i+i_len, j:j+j_len] = D_ij

        # make symmetric
        D = np.triu(D) + np.triu(D, 1).T

        return D

    def compute_eigenvectors(self, D):
        args = self.args
        try:
            dmap = dmaps.DiffusionMap(D)
            eps = self.tune_epsilon(dmap, osp.join(args.odir_i, 'tuning.png'))
            dmap.set_kernel_bandwidth(eps)
            dmap.compute(5, 0.5)
        except Exception as e:
            print(e, file = args.log_file)
            return None, None

        v = dmap.get_eigenvectors()
        w = dmap.get_eigenvalues()
        print('w', w, file = args.log_file)
        order = np.argsort(v[:, 1])
        if not args.experimental:
            print('\norder corr: ',
                pearson_round(order, np.arange(0, args.N, 1), stat = 'spearman'),
                file = args.log_file)

        return v, order

    def tune_epsilon(self, input, ofile):
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

        eps_final = np.exp(np.mean(slice_X))
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

    def xyz_diffusion(self):
        args = self.args
        self.xyz = self.xyz.reshape(N, m * 3)
        args.odir_i = args.odir

        # compute distance
        D = self.compute_distance(input = xyz, metric = dmaps.metrics.rmsd)

        # compute eigenvectors
        v, _ = self.compute_eigenvectors(D)

        # plot
        plot_eigenvectors(v, xyz.reshape(-1, m, 3), args.odir)

    def contact_diffusion(self):
        args = self.args

        order = None
        odir_prev = None
        updater = Updater(args.k, args.log_file)
        for it in range(args.its+1):
            args.odir_i = osp.join(args.scratch_dir, f'iteration_{it}')
            os.mkdir(args.odir_i, mode = 0o755)
            print(f"Iteration {it}")
            print(f"Iteration {it}", file = args.log_file)
            if it == 0:
                # apply gaussian
                GP = PreProcessing(args, self.files)
                GP.process(args.jobs, args.log_file)
            else:
                # diag processing
                if args.metric == 'scc' or args.metric == 'inner_product':
                    input = osp.join(odir_prev, 'sc_contacts')
                else:
                    diag_dir = osp.join(args.odir_i, 'sc_contacts_diag')
                    diag_processsing_chunk(osp.join(odir_prev, 'sc_contacts'),
                                    diag_dir, args)
                    input = diag_dir

                # compute distance
                D = self.compute_distance(input, args.metric)
                if D is None:
                    break

                # compute eigenvectors
                v, order = self.compute_eigenvectors(D)
                if v is None:
                    break

                np.savetxt(osp.join(args.odir_i, 'order.txt'), order, fmt='%i')
                if args.plot:
                    plot_eigenvectors(v, self.xyz, args.odir_i)

                # update
                if it == args.its:
                    # skip update on last iteration
                    continue
                if args.update_mode == 'eig':
                    sc_contacts = updater.update_eig_chunk(v[:, 1],
                                            osp.join(odir_prev, 'sc_contacts'),
                                            args.odir_i)
                elif args.update_mode == 'knn':
                    sc_contacts = updater.update_kNN_chunk(v[:,1:4],
                                            osp.join(odir_prev, 'sc_contacts'),
                                            args.odir_i)

            # Plots
            if args.plot:
                t0 = time.time()
                plot_contacts(osp.join(args.odir_i, 'sc_contacts'), order, args)
                tf = time.time()
                print_time(t0, tf, 'plot', file = args.log_file)
            print('\n', file = args.log_file)
            odir_prev = args.odir_i

    def contact_laplacian(self):
        args = self.args

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
                eps_final = self.tune_epsilon(D, ofile = osp.join(args.odir_i, 'tuning.png'))
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
                if args.plot:
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

def cleanup(args):
    for it in range(args.its+1):
        diag_dir = osp.join(args.scratch_dir, f'iteration_{it}', 'sc_contacts_diag')
        if osp.exists(diag_dir):
            rmtree(diag_dir)

    # move files from scratch to odir
    move(args.scratch_dir, args.odir)

def main():
    args = getArgs()

    if args.mode == 'xyz_diffusion':
        Diffusion(args).xyz_diffusion()
    elif args.mode == 'contact_diffusion':
        diff = Diffusion(args, True)
        diff.contact_diffusion()
    elif args.mode == 'contact_laplacian':
        Diffusion(args).contact_laplacian()

    cleanup(args)



if __name__ == '__main__':
    main()
    # plot_gif_michrom()
