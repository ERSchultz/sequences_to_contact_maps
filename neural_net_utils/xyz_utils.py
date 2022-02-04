import os.path as osp
import sys
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)

import numpy as np
import csv
import json
import math
from collections import defaultdict
import time
from numba import jit, njit

import matplotlib.pyplot as plt

from utils import LETTERS, load_X_psi

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from plotting_functions import plotContactMap

def xyzWrite(xyz, outfile, writestyle, comment = '', x = None):
    '''
    Write the coordinates of all particle to a file in .xyz format.
    Inputs:
        xyz: shape (T, N, 3) or (N, 3) array of all particle positions (angstroms)
        outfile: name of file
        writestyle: 'w' (write) or 'a' (append)
    '''
    if len(xyz.shape) == 3:
        T, N, _ = xyz.shape
        for i in range(T):
            xyzWrite(xyz[i, :, :], outfile, 'a', comment = comment, x = x)
    else:
        N = len(xyz)
        with open(outfile, writestyle) as f:
            f.write('{}\n{}\n'.format(N, comment))
            for i in range(N):
                f.write(f'{i} {xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n')

def xyzLoad(xyz_filepath, delim = '\t', multiple_timesteps=False):
    xyz = []
    with open(xyz_filepath, 'r') as f:
        N = int(f.readline())
        reader = csv.reader(f, delimiter = delim)
        xyz_timestep = []
        for line in reader:
            if len(line) > 1:
                i = int(line[0])
                xyz_i = [float(j) for j in line[1:4]]
                xyz_timestep.append(xyz_i)
                if i == N-1:
                    xyz.append(xyz_timestep)
                    xyz_timestep=[]

    xyz = np.array(xyz)
    if not multiple_timesteps:
        xyz = xyz[0]
    return xyz

def findLabelCentroid(xyz, psi):
    '''
    Finds center of mass for each label in psi.

    Inputs:
        xyz: np array of shape m x 3
        psi: np array of shape m x k

    Output:
        centroids: dictionary of centroids (keys are captial letters)
    '''
    m, k = psi.shape

    centroids = {}
    for i, letter in enumerate(LETTERS[:k]):
        xyz_i = xyz[psi[:, i] == 1, :]
        centroid = np.mean(xyz_i, axis = 0)
        centroids[letter] = centroid

    return centroids

def findDistanceBetweenCentroids(centroids):
    '''Computes distance between pairs of centroids (dict).'''
    k = len(centroids.keys())
    distances = np.zeros((k,k))
    for i, letter_i in enumerate(LETTERS[:k]):
        centroid_i = centroids[letter_i]
        for j in range(i):
            letter_j = LETTERS[j]
            centroid_j = centroids[letter_j]
            dist = np.linalg.norm(centroid_i - centroid_j)
            distances[i,j] = dist
            distances[j,i] = dist

    return distances

def xyz_to_contact_grid(xyz, grid_size):
    '''
    Converts xyz to contact map via grid.

    Inputs:
        xyz: np array of shape N, m, 3 (N is optional)
        grid_size: size of grid (nm)
    '''
    if len(xyz.shape) == 3:
        N, m, _ = xyz.shape
    else:
        N = 1
        m, _ = xyz.shape
        xyz = xyz.reshape(-1, m, 3)

    contact_map = np.zeros((m,m))
    for n in range(N):
        # use dictionary to find contacts
        grid_dict = defaultdict(list) # grid (x, y, z) : bead id list
        for i in range(m):
            grid_i = tuple([d // grid_size for d in xyz[n, i, :]])
            grid_dict[grid_i].append(i)

        for bead_list in grid_dict.values():
            for i in bead_list:
                for j in bead_list:
                    contact_map[i,j] += 1
                    contact_map[j,i] += 1

    return contact_map

# @njit
def xyz_to_contact_distance(xyz, cutoff_distance):
    '''
    Converts xyz to contact map via grid
    '''
    if len(xyz.shape) == 3:
        N = xyz.shape[0]
        m = xyz.shape[1]
    else:
        N = 1
        m = xyz.shape[0]
        xyz = xyz.reshape(-1, m, 3)

    contact_map = np.zeros((m,m))
    for n in range(N):
        for i in range(m):
            for j in range(i+1):
                dist = np.linalg.norm(xyz[n, i, :] -xyz[n, j, :])
                if dist <= cutoff_distance:
                    contact_map[i,j] += 1
                    contact_map[j,i] += 1

    return contact_map

def main():
    dir='/home/eric/dataset_test/samples/sample82'
    file = osp.join(dir, 'data_out/output.xyz')

    config_file = osp.join(dir, 'config.json')
    with open(config_file, 'rb') as f:
        config = json.load(f)
        grid_size = int(config['grid_size'])


    x = np.load(osp.join(dir, 'x.npy'))
    y = np.load(osp.join(dir, 'y.npy'))
    xyz = xyzLoad(file, multiple_timesteps=True)
    N, m, _ = xyz.shape
    N = 5


    t0 = time.time()
    overall = xyz_to_contact_grid(xyz[:N], grid_size)
    tf = time.time()
    print(tf - t0)
    plotContactMap(overall, osp.join(dir, 'sc_contact', 'overall.png'))

    t0 = time.time()
    overall2 = xyz_to_contact_distance(xyz[:N], grid_size)
    tf = time.time()
    print(tf - t0)
    plotContactMap(overall2, osp.join(dir, 'sc_contact', 'overall2.png'))


    # dif = overall - y
    # plotContactMap(dif, osp.join(dir, 'sc_contact', 'dif.png'), cmap = 'blue-red')

    # print(np.array_equal(y, overall))

def main2():
    dir = '/home/eric/dataset_test/samples'

    for sample in range(90, 97):
        if sample != 92:
            continue
        sample_dir = osp.join(dir, f'sample{sample}')
        xyz = xyzLoad(osp.join(sample_dir, 'data_out', 'output.xyz'), multiple_timesteps = True)
        N, _, _ = xyz.shape
        _, psi = load_X_psi(osp.join(dir, 'sample83'))
        _, k = psi.shape
        distances = np.zeros((N, k, k))
        for i in range(N):
            centroids = findLabelCentroid(xyz[i], psi)
            distances_i = findDistanceBetweenCentroids(centroids)
            distances[i, :, :] = distances_i

        plt.hist(distances[:, 0, 2])
        plt.savefig(osp.join(sample_dir, 'AC_dist.png'))
        plt.close()

        plt.scatter(distances[:, 0, 2], np.linspace(0, N, N))
        plt.xlabel('A-B distance')
        plt.ylabel('sample index')
        plt.savefig(osp.join(sample_dir, 'AC_dist_vs_i.png'))
        plt.close()

        # y_grid = xyz_to_contact_grid(xyz, 28.7)
        # np.savetxt(osp.join(sample_dir, 'y_grid.txt'), y_grid)
        # plotContactMap(y_grid, osp.join(sample_dir, 'y_grid.png'), vmax = 'mean')
        # y_dist = xyz_to_contact_distance(xyz, 28.7)
        # plotContactMap(y_dist, osp.join(sample_dir, 'y_dist.png'), vmax = 'mean')

        y_600_800 = xyz_to_contact_grid(xyz[600:800], 28.7)
        np.savetxt(osp.join(sample_dir, 'y_600_800.txt'), y_600_800)
        plotContactMap(y_600_800, osp.join(sample_dir, 'y_600_800.png'), vmax = 'mean')

        y_100_300 = xyz_to_contact_grid(xyz[100:300], 28.7)
        np.savetxt(osp.join(sample_dir, 'y_100_300.txt'), y_100_300)
        plotContactMap(y_100_300, osp.join(sample_dir, 'y_100_300.png'), vmax = 'mean')

        y_200 = xyz_to_contact_distance(xyz[200], 28.7)
        np.savetxt(osp.join(sample_dir, 'y_200.txt'), y_200)
        plotContactMap(y_200, osp.join(sample_dir, 'y_200.png'), vmax = 'max')

        y_700 = xyz_to_contact_distance(xyz[700], 28.7)
        np.savetxt(osp.join(sample_dir, 'y_700.txt'), y_700)
        plotContactMap(y_700, osp.join(sample_dir, 'y_700.png'), vmax = 'max')






if __name__ == '__main__':
    main2()
