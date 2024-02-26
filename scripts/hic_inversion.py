import json
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pylib.analysis as analysis
from pylib.Pysim import Pysim
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_D
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.similarity_measures import SCC


def normalize(arr):
    arr = arr / np.mean(arr.diagonal())
    # arr = arr / np.mean(arr)
    arr[arr>1] = 1
    return arr

def S_to_Y():
    dir = '/home/erschultz/dataset_12_06_23/samples/sample1'
    grid_root = 'optimize_grid_b_200_v_8_spheroid_1.5'
    y_bonded = np.load(osp.join(dir, grid_root, 'y.npy'))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_bonded, 'freq')
    y_bonded = calculate_D(meanDist)
    y_bonded = normalize(y_bonded)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_bonded, 'freq')
    plt.plot(meanDist, label='y_bonded')

    scc = SCC(h=1, K=100)

    me_dir = osp.join(dir, f'{grid_root}-max_ent10')
    S_me = np.load(osp.join(me_dir, 'iteration30/S.npy'))

    gnn_dir = osp.join(dir, f'{grid_root}-GNN629')
    odir = osp.join(gnn_dir, 'test')
    if not osp.exists(odir):
        os.mkdir(odir)

    y = np.load(osp.join(gnn_dir, 'y.npy'))
    y = normalize(y)
    plot_matrix(y, osp.join(odir, 'y.png'), vmax = 'mean')
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'freq')
    plt.plot(meanDist, label='y')


    S = np.load(osp.join(gnn_dir, 'S.npy'))
    print('S', scc.scc(S, S_me))
    print('exp(S)', scc.scc(np.exp(-S), np.exp(-S_me)))
    print('log(S)', scc.scc(np.sign(S) * np.log(np.abs(S) + 1),
                            np.sign(S_me) * np.log(np.abs(S_me) + 1)))

    y_S = np.exp(-S)
    y_S = normalize(y_S)
    print('y_S', scc.scc(y, y_S))
    plot_matrix(y_S, osp.join(odir, 'y_S.png'), vmax = 'max')

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_S, 'freq')
    plt.plot(meanDist, label='exp(-S)')

    # y_add = y_bonded + y_S
    # y_add /= np.mean(y_add.diagonal())
    # print('add', scc.scc(y, y_add))
    # plot_matrix(y_add, osp.join(odir, 'y_add.png'), vmax = 'mean')
    #
    y_mul = y_bonded * y_S
    y_mul = normalize(y_mul)
    print('mul', scc.scc(y, y_mul))
    plot_matrix(y_mul, osp.join(odir, 'y_mul.png'), vmax = 'max')
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_mul, 'freq')
    plt.plot(meanDist, label='exp(-S)*y_bonded')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('s')
    plt.ylabel('P(s)')
    plt.ylim(1e-7, None)
    plt.savefig(osp.join(odir, 'p_s.png'))

def Y_to_G():
    dir = '/home/erschultz/dataset_12_06_23/samples/sample1'
    grid_root = 'optimize_grid_b_200_v_8_spheroid_1.5'
    gnn_dir = osp.join(dir, f'{grid_root}-GNN629')
    odir = osp.join(gnn_dir, 'test')
    if not osp.exists(odir):
        os.mkdir(odir)

    y = np.load(osp.join(gnn_dir, 'y.npy'))
    np.save(osp.join(odir, 'y.npy'), y)
    y = normalize(y)

    y_bonded = np.load(osp.join(dir, grid_root, 'y.npy'))
    meandist = DiagonalPreprocessing.genomic_distance_statistics(y_bonded, 'freq')
    y_bonded = calculate_D(meandist)
    y_bonded = normalize(y_bonded)
    plot_matrix(y_bonded, osp.join(odir, 'y_bonded.png'), vmax = 'mean', title='Y_bonded')



    S = np.load(osp.join(gnn_dir, 'S.npy'))
    plot_matrix(S, osp.join(odir, 'S.png'), vmax = 'max', vmin='center', cmap='bluered', title='S')

    G = -np.log(y / y_bonded)
    np.save(osp.join(odir, 'G.npy'), G)
    plot_matrix(G, osp.join(odir, 'G.png'), vmax = 'max', vmin='center', cmap='bluered', title='G')
    np.savetxt(osp.join(odir, 'gmatrix.txt'), G)

    plt.scatter(G, S, alpha=0.01)
    plt.xlabel(r'$G_{ij}$', fontsize=16)
    plt.ylabel(r'$S_{ij}$', fontsize=16)
    plt.axline((0,0), slope = 1, c='k')
    plt.savefig(osp.join(odir, 'scatter.png'))
    plt.close()

    plt.scatter(y, S, alpha=0.01)
    plt.xscale('log')
    plt.xlabel(r'$H_{ij}$', fontsize=16)
    plt.ylabel(r'$S_{ij}$', fontsize=16)
    # plt.axline((0,0), slope = 1, c='k')
    plt.savefig(osp.join(odir, 'scatter2.png'))
    plt.close()

    plt.scatter(y, np.exp(-S), alpha=0.01)
    plt.xscale('log')
    plt.xlabel(r'$H_{ij}$', fontsize=16)
    plt.ylabel(r'exp($-S_{ij}$)', fontsize=16)
    # plt.axline((0,0), slope = 1, c='k')
    plt.savefig(osp.join(odir, 'scatter3.png'))
    plt.close()

def simulation():
    dir = '/home/erschultz/dataset_12_06_23/samples/sample1/optimize_grid_b_200_v_8_spheroid_1.5-GNN629/test/G_simulation'
    y_file = osp.join(dir, 'y.npy')
    y = np.load(y_file).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    np.fill_diagonal(y, 1)
    m = len(y)

    with open(osp.join(dir, 'config.json')) as f:
        config = json.load(f)

    S = np.loadtxt(osp.join(dir, 'gmatrix.txt'))

    stdout = sys.stdout
    with open(osp.join(dir, 'log.log'), 'w') as sys.stdout:
        # sim = Pysim(dir, config, None, y, randomize_seed = True,
        #             mkdir = False, smatrix = S)
        # t = sim.run_eq(10000, 300000, 1)
        # print(f'Simulation took {np.round(t, 2)} seconds')

        analysis.main_no_maxent(dir=dir)
    sys.stdout = stdout

if __name__ == '__main__':
    Y_to_G()
    # S_to_Y()
    # simulation()
