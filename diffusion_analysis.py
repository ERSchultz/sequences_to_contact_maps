import os
import os.path as osp
import time

import dmaps  # https://github.com/hsidky/dmaps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.plotting_utils import plot_matrix
from utils.utils import pearson_round, print_time
from utils.xyz_utils import xyz_load


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

def main():
    dir = '/home/eric/dataset_test/samples/sample91'
    odir = osp.join(dir, 'diffusion_test')
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    # load xyz
    xyz = xyz_load(osp.join(dir, 'data_out/output.xyz'),
                    multiple_timesteps=True, save = True, N_min = 10,
                    N_max = None, down_sampling = 1)
    N, m, _ = xyz.shape
    xyz = xyz.reshape(N, m * 3)

    # compute distance
    t0 = time.time()
    dist = dmaps.DistanceMatrix(xyz)
    dist.compute(metric=dmaps.metrics.rmsd)
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


    # Plot result. Scale by top eigenvector.
    v = dmap.get_eigenvectors()
    w = dmap.get_eigenvalues()
    print('w', w)
    order = np.argsort(v[:, 1])
    print('\n\norder corr: ', pearson_round(order, np.arange(0, N, 1), stat = 'spearman'))
    sc = plt.scatter(v[:,1]/v[:,0], v[:,2]/v[:,0], c = np.arange(0, N, 1))
    plt.colorbar(sc)
    plt.xlabel(r'$v_2$', fontsize = 16)
    plt.ylabel(r'$v_3$', fontsize = 16)
    plt.savefig(osp.join(odir, 'projection.png'))
    plt.close()


if __name__ == '__main__':
    main()
