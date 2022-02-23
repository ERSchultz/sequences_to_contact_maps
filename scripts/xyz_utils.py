import csv
import json
import os.path as osp
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .plotting_functions import plotContactMap
from .utils import LETTERS


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







if __name__ == '__main__':
    main()
