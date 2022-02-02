import os.path as osp
import sys
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)

import numpy as np
import csv

import matplotlib.pyplot as plt

from utils import LETTERS, load_X_psi

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


def main():
    dir = '/home/eric/dataset_test/samples'

    for sample in range(90, 97):
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




if __name__ == '__main__':
    main()
