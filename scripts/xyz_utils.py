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
from pylib.utils.utils import LETTERS, print_time
from pylib.utils.xyz import *
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import nan_euclidean_distances


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
