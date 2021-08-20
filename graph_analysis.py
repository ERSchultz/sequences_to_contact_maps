import os
import os.path as osp

import numpy as np
import networkx as nx
import argparse
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plotDegreeDist(graph, title = None, ofile = None, weighted = False):
    if weighted:
        degrees = [graph.degree(n, 'weight') for n in graph.nodes()]
    else:
        degrees = [graph.degree(n) for n in graph.nodes()]
    plt.hist(degrees)
    if title is not None:
        plt.title(title)
    if ofile is not None:
        plt.savefig(ofile)
    else:
        plt.show()

def getPCVarianceExplainedRatio(y, k = 5, standardize = False):
    '''Returns variance explained ratio by top k principal components'''
    pca = PCA()
    if standardize:
        scaler = StandardScaler()
        y = scaler.fit_transform(y.copy())
    pca.fit(y)

    return pca.explained_variance_ratio_[:k]


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    # '../../sequences_to_contact_maps/dataset_04_18_21'
    # "/project2/depablo/erschultz/dataset_04_18_21"
    parser.add_argument('--data_folder', type=str, default='dataset_04_18_21', help='Location of input data')
    parser.add_argument('--sample', type=int, default=2, help='sample id')

    args = parser.parse_args()
    args.sample_folder = osp.join(args.data_folder, 'samples', 'sample{}'.format(args.sample))
    return args


def main():
    args = getArgs()
    y_diag = np.load(osp.join(args.sample_folder, 'y_diag.npy'))
    num_PCs = getPCVarianceExplainedRatio(y_diag)
    print(num_PCs)


    # ypos = y_diag.copy()
    # ypos[y_diag < 1] = 0
    # pos_graph = nx.convert_matrix.from_numpy_matrix(ypos)
    # plot_degree_dist(pos_graph, title ='pos edges')
    #
    # yneg = y_diag.copy()
    # yneg[y_diag > 1] = 0
    #
    # neg_graph = nx.convert_matrix.from_numpy_matrix(yneg)
    # plot_degree_dist(neg_graph, title ='neg edges')

if __name__ == '__main__':
    main()
