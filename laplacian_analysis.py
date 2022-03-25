import os
import os.path as osp
import time

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances, silhouette_score
from sympy import solve, symbols
from utils.load_utils import load_sc_contacts
from utils.plotting_utils import (plot_matrix, plot_matrix_gif,
                                  plot_sc_contact_maps_inner)
from utils.utils import (diagonal_preprocessing_bulk,
                         genomic_distance_statistics, pearson_round,
                         print_size, print_time, triu_to_full)
from utils.xyz_utils import find_dist_between_centroids, find_label_centroid


def sort_laplacian(A_tilde, xyz, sc_contacts, odir, it):
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
    where = np.argsort(v[:, i])

    plot_eigenvectors(v, xyz, sc_contacts, odir, it)

    # sort sc_contacts
    sc_contacts = sc_contacts[where, :]

    # merge adjacent contacts
    # modify in place to preserve RAM
    # I did test this - it works
    # N, _ = sc_contacts.shape
    # for i in range(N):
    #     if i == 0:
    #         new = np.mean(sc_contacts[0:2, :], axis = 0)
    #     elif i == N:
    #         new = np.mean(sc_contacts[-2:N, :], axis = 0)
    #     else:
    #         new = np.mean(sc_contacts[i-1:i+2, :], axis = 0)
    #     if i > 0:
    #         sc_contacts[i-1, :] = new_prev
    #     new_prev = new.copy()
    # sc_contacts[N-1] = new_prev

    # plot sorted contacts as gif

    # undo sort
    order = np.argsort(where)
    sc_contacts = sc_contacts[order]

    # save order
    np.savetxt(osp.join(odir, f'order_{i}.txt'), where, fmt='%i')

    return sc_contacts, where

def plot_eigenvectors(v, xyz, sc_contacts, odir, it):
    N = len(v)

    # plot first 2 nonzero eigenvectors, color by order
    sc = plt.scatter(v[:, 1]/v[:, 0], v[:, 2]/v[:, 0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_2$', fontsize = 16)
    plt.ylabel(r'$v_3$', fontsize = 16)
    plt.savefig(osp.join(odir, f'projection23_{it}.png'))
    plt.close()

    # plot 3 and 4 eigenvectors, color by order
    sc = plt.scatter(v[:,2]/v[:,0], v[:,3]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.savefig(osp.join(odir, f'projection34_{it}.png'))
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
    plt.savefig(osp.join(odir, f'projection234_{it}.png'))
    plt.close()


    # centroid distance
    dist = np.zeros(N)
    psi = np.zeros((1500, 3))
    psi[0:200, 0] = 1
    psi[200:1300, 1] = 1
    psi[1300:, 2] = 1
    for i in range(N):
        centroids = find_label_centroid(xyz[i].reshape(-1, 3), psi)
        distances = find_dist_between_centroids(centroids)
        dist[i] = distances[0, 2]

    print('\n\ncentroid corr: ', pearson_round(dist, v[:, 1], stat = 'pearson'))


    # plot first 2 nonzero eigenvectors, color by distance
    sc = plt.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], c = dist)
    plt.colorbar(sc)
    plt.xlabel(r'$v_2$', fontsize = 16)
    plt.ylabel(r'$v_3$', fontsize = 16)
    plt.savefig(osp.join(odir, f'projection23_dist_{it}.png'))
    plt.close()

    # plot 3 and 4 eigenvectors, color by distance
    sc = plt.scatter(v[:,2]/v[:,0], v[:,3]/v[:,0], c = dist)
    plt.colorbar(sc)
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.savefig(osp.join(odir, f'projection34_dist_{it}.png'))
    plt.close()

    # k_means
    num_vecs = 3
    X = v[:,1:1+num_vecs]
    sil_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k).fit(X)
        sil_scores.append(silhouette_score(X, kmeans.labels_))
    plt.plot(np.arange(2, 10, 1), sil_scores)
    plt.savefig(osp.join(odir, f'k_means_sil_score_{it}.png'))
    plt.close()
    k = np.argmax(sil_scores) + 2
    print(f'Using k = {k}')
    kmeans = KMeans(n_clusters=k).fit(X)

    # plot average contact map within cluster
    for cluster in range(k):
        ind = np.argwhere(kmeans.labels_ == cluster)
        y_cluster = triu_to_full(np.sum(sc_contacts[ind.reshape(-1), :], axis = 0))
        sum_cluster = np.sum(y_cluster[0:200, 1300:1500])/np.max(y_cluster)
        print(sum_cluster)
        plot_matrix(y_cluster, osp.join(odir, f'cluster{cluster}_contacts_{it}.png'),
                    vmax = 'max', title = sum_cluster)

    # plot first 2 nonzero eigenvectors, color by order
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
    plt.savefig(osp.join(odir, f'projection23_kmeans_{it}.png'))
    plt.close()

    # plot 3 and 4 eigenvectors, color by order
    for cluster, c in zip(range(k), colors):
        ind = np.argwhere(kmeans.labels_ == cluster)
        plt.scatter(v[ind, 2]/v[ind, 0], v[ind, 3]/v[ind, 0], color = c['color'],
                    label = cluster)
    plt.xlabel(r'$v_3$', fontsize = 16)
    plt.ylabel(r'$v_4$', fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(odir, f'projection34_kmeans_{it}.png'))
    plt.close()

    # plot 2,3,4 eigenvectors, color by order
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
    plt.savefig(osp.join(odir, f'projection234_kmeans_{it}.png'))
    plt.close()

def tune_epsilon(A, ofile):
    X = np.arange(-8, 5, 1)
    epsilons = np.exp(X)
    Y = np.zeros_like(epsilons)
    for i, eps in enumerate(epsilons):
        A_eps = np.exp(-1/2 * A**2 / eps)
        Y[i] = np.sum(A_eps)

    # find best linear regression on subset of 3/4 points
    best_indices = (-1, -1)
    best_score = 0
    best_reg = None
    for size in [3, 4]:
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

    slice_X = X[best_indices[0]:best_indices[1]]
    slice_X_big = X[best_indices[0]-1:best_indices[1]+1]
    slice_Y = Y[best_indices[0]:best_indices[1]]
    coef = np.round(best_reg.coef_[0], 3)
    intercept = np.round(best_reg.intercept_, 3)

    eps_final = np.round(np.exp(np.mean(slice_X)), 3)
    print(f'Using epsilon = {eps_final}')

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

    return np.exp(-1/2 * A**2 / eps_final)

def main_sc():
    dir = '/home/erschultz/dataset_test/samples/sample92'
    odir = osp.join(dir, 'sc_contact')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    overall = np.load(osp.join(dir, 'y.npy'))
    mean_per_diag = genomic_distance_statistics(overall, mode = 'prob')
    sc_contacts, xyz = load_sc_contacts(dir, N_max = None, triu = True,
                                    gaussian = True, zero_diag = True, jobs = 8,
                                    down_sampling = 3, sparsify = False,
                                    return_xyz = True)
    sc_contacts_orig = sc_contacts.copy()
    print_size(sc_contacts, 'sc_contacts')
    N, _ = sc_contacts.shape
    for i in range(1):
        # diag processing
        t0 = time.time()
        sc_contacts_diag = diagonal_preprocessing_bulk(sc_contacts, mean_per_diag,
                                                        triu = True)
        tf = time.time()
        print_time(t0, tf, 'diag')

        # distance (using diag)
        t0 = time.time()
        distances = pairwise_distances(sc_contacts_diag, sc_contacts_diag,
                                        metric = 'correlation')
        plot_matrix(distances, ofile = osp.join(odir, f'distances{i}.png'),
                        vmin = 'min', vmax = 'max')
        tf = time.time()
        print_time(t0, tf, '\n\ndistances')

        # Adjacency
        A = tune_epsilon(distances, ofile = osp.join(odir, f'tuning_{i}.png'))
        # A = 1 - distances / np.max(distances)
        plot_matrix(A, ofile = osp.join(odir, f'A{i}.png'),
                        vmin = 'min', vmax = 'max')
        np.savetxt(osp.join(odir, f'A_{i}.txt'), A, fmt='%.2f')

        # Laplacian
        t0 = time.time()
        A_tilde = laplacian(A, normed = True)
        plot_matrix(A_tilde, ofile = osp.join(odir, f'A_tilde_{i}.png'),
                        vmin = np.min(A_tilde), vmax = np.max(A_tilde),
                        cmap = 'blue-red')
        np.savetxt(osp.join(odir, f'A_tilde_{i}.txt'), A_tilde, fmt='%.2f')
        sc_contacts, order = sort_laplacian(A_tilde, xyz, sc_contacts, odir, i)
        print('\n\norder corr: ', pearson_round(order, np.arange(0, N, 1),
                                                stat = 'spearman'))
        tf = time.time()
        print_time(t0, tf, 'laplacian')

        # # Plots
        # t0 = time.time()
        # plot_sc_contact_maps_inner(sc_contacts, osp.join(odir, f'iteration_{i}'),
        #                             count = 5, jobs = 6)
        # tf = time.time()
        # print_time(t0, tf, 'plot')
        # print('\n')




if __name__ == '__main__':
    main_diff()
