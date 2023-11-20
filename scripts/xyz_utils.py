'''Functions for manipulation .xyz files.'''

import csv
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from pylib.utils.xyz import *
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import nan_euclidean_distances

from .utils import LETTERS, print_time


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

# @njit
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
