import csv
import json
import os.path as osp
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy.sparse import csr_array

from .utils import LETTERS, print_time


def xyz_write(xyz, outfile, writestyle, comment = '', x = None):
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
            xyz_write(xyz[i, :, :], outfile, 'a', comment = comment, x = x)
    else:
        N = len(xyz)
        with open(outfile, writestyle) as f:
            f.write('{}\n{}\n'.format(N, comment))
            for i in range(N):
                f.write(f'{i} {xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n')

def xyz_load(xyz_filepath, delim = '\t', multiple_timesteps = False, save = False,
            N_min = None, N_max = None, down_sampling = 1):
    t0 = time.time()
    xyz_npy_file = osp.join(osp.split(xyz_filepath)[0], 'xyz.npy')
    if osp.exists(xyz_npy_file):
        xyz = np.load(xyz_npy_file)
    else:
        xyz = []
        with open(xyz_filepath, 'r') as f:
            N = int(f.readline())
            reader = csv.reader(f, delimiter = delim)
            xyz_timestep = np.empty((N, 3))
            for line in reader:
                if len(line) > 1:
                    i = int(line[0])
                    xyz_timestep[i, :] = [float(j) for j in line[1:4]]
                    if i == N-1:
                        xyz.append(xyz_timestep)
                        xyz_timestep=np.empty((N, 3))

        xyz = np.array(xyz)
        if save:
            np.save(xyz_npy_file, xyz)
    if not multiple_timesteps:
        xyz = xyz[0]
    if N_min is None:
        N_min = 0
    if N_max is None:
        N_max = len(xyz)
    xyz = xyz[N_min:N_max:down_sampling]
    tf = time.time()
    print(f'Loaded xyz with shape {xyz.shape}')
    print_time(t0, tf, 'xyz load')
    return xyz

def lammps_load(filepath, save = False, N_min = None, N_max = None, down_sampling = 1):
    xyz_npy_file = osp.join(osp.split(filepath)[0], 'xyz.npy')
    x_npy_file = osp.join(osp.split(filepath)[0], 'x.npy')
    t0 = time.time()
    if osp.exists(xyz_npy_file):
        xyz = np.load(xyz_npy_file)
    else:
        xyz = []
        with open(filepath, 'r') as f:
            line = 'null'
            while line != '':
                line = f.readline().strip()
                if line == 'ITEM: NUMBER OF ATOMS':
                    N = int(f.readline().strip())
                    xyz_timestep = np.empty((N, 3))

                if line == 'ITEM: ATOMS id type xu yu zu':
                    line = f.readline().strip().split(' ')
                    while line[0].isnumeric():
                        i = int(line[0]) - 1
                        xyz_timestep[i, :] = [float(j) for j in line[2:5]]
                        if i == N-1:
                            xyz.append(xyz_timestep)
                        line = f.readline().strip().split(' ')
        xyz = np.array(xyz)
        if save:
            np.save(xyz_npy_file, xyz)

    if osp.exists(x_npy_file):
        x = np.load(x_npy_file)
    else:
        x = []
        with open(filepath, 'r') as f:
            keep_reading = True
            while keep_reading:
                line = f.readline().strip()
                if line == 'ITEM: ATOMS id type xu yu zu':
                    keep_reading = False
                    line = f.readline().strip().split(' ')
                    while line[0].isnumeric():
                        i = int(line[0]) - 1
                        x.append(int(line[1])-1)
                        line = f.readline().strip().split(' ')
        N = len(x)
        x_arr = np.zeros((N, np.max(x)+1))
        x_arr[np.arange(N), x] = 1
        if save:
            np.save(x_npy_file, x_arr)

    if N_min is None:
        N_min = 0
    if N_max is None:
        N_max = len(xyz)
    xyz = xyz[N_min:N_max:down_sampling]
    tf = time.time()
    print(f'Loaded xyz with shape {xyz.shape}')
    print_time(t0, tf, 'xyz load')
    return xyz, x_arr

def find_label_centroid(xyz, psi):
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

def find_dist_between_centroids(centroids):
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

def xyz_to_contact_grid(xyz, grid_size, sparse_format = False, dtype = np.int32):
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
        m, d = xyz.shape
        xyz = xyz.reshape(1, m, d)

    contact_map = np.zeros((m, m)).astype(dtype)
    for n in range(N):
        # use dictionary to find contacts
        grid_dict = defaultdict(list) # grid (x, y, z) : bead id list
        for i in range(m):
            grid_i = tuple([d // grid_size for d in xyz[n, i, :3]])
            grid_dict[grid_i].append(i)

        for bead_list in grid_dict.values():
            for i in bead_list:
                for j in bead_list:
                    contact_map[i,j] += 1

    if sparse_format:
        return csr_array(contact_map)

    return contact_map

# @njit
def xyz_to_contact_distance(xyz, cutoff_distance, verbose = False):
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
    t0 = time.time()
    for n in range(N):
        if verbose:
            prcnt_done = n/N * 100
            t = time.time() - t0
            if prcnt_done % 5 == 0:
                print(f'{prcnt_done}%')
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
    xyz = xyz_load(file, multiple_timesteps=True)
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
