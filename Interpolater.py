import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import scipy.stats as ss

from utils.plotting_utils import plot_matrix
from utils.utils import DiagonalPreprocessing, rescale_matrix


class Interpolater():
    def __init__(self, dataset, sample, methods):
        self.dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        self.odir = osp.join(self.dir, 'Interpolation')
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)
        self.odir = osp.join(self.odir, "_".join(methods))
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)
        self.ofile = open(osp.join(self.odir, 'log.log'), 'a')

        self.y = np.load(osp.join(self.dir, 'y.npy'))

        with open(osp.join(self.dir, 'import.log'), 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split('=')
                if line[0] == 'chrom':
                    self.chrom = line[1]
                elif line[0] == 'start':
                    self.start = int(line[1])
                elif line[0] == 'end':
                    self.end = int(line[1])
                elif line[0] == 'resolution':
                    self.res = int(line[1])

        crop_left = 1130
        crop_right = 1200
        plot_matrix(self.y[crop_left:crop_right, crop_left:crop_right],
                    osp.join(self.odir, 'y_crop.png'), vmax = 'mean')
        y_pool = rescale_matrix(self.y, 5)
        # plot_matrix(y_pool[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, 'y_pool_crop.png'), vmax = 'mean')
        plot_matrix(y_pool, osp.join(self.odir, 'y_pool.png'), vmax = 'mean')

        self.interp_locations = set()
        for method in methods:
            method_split = method.split('-')
            method = method_split[0]
            if len(method_split) > 1:
                cutoff = float(method_split[1])
            else:
                cutoff = None
            if method == 'zeros':
                self.find_zeros_along_diagonal()
            elif method == 'sum_mzscore':
                self.find_sum_outliers(cutoff, modified = True)
            elif method == 'sum_zscore':
                self.find_sum_outliers(cutoff)
            elif method == 'mappability':
                self.find_low_mappability(cutoff)
            elif method == 'sparsity':
                self.find_high_sparsity(cutoff)
            elif method == 'sparsity_mzscore':
                self.find_sparsity_outliers(cutoff, modified = True)
            elif method == 'sparsity_zscore':
                self.find_sparsity_outliers(cutoff)

            y_interp = self.linear_interpolate(fill_mean = False)

        # plot_matrix(self.y, osp.join(self.odir, f'y_lines_{"_".join(methods)}.png'),
                    # vmax = 'mean', lines = self.interp_locations)
        # plot_matrix(y_interp, osp.join(self.odir, f'y_{"_".join(methods)}.png'), vmax = 'mean')
        plot_matrix(y_interp[crop_left:crop_right, crop_left:crop_right],
                    osp.join(self.odir, f'y_crop_{"_".join(methods)}.png'), vmax = 'mean')
        y_interp_pool = rescale_matrix(y_interp, 5)
        plot_matrix(y_interp_pool, osp.join(self.odir, f'y_pool_{"_".join(methods)}.png'), vmax = 'mean')
        # plot_matrix(y_interp_pool[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, f'y_pool_crop_{"_".join(methods)}.png'), vmax = 'mean')

        np.save(osp.join(self.odir, f'y_interpolate_{"_".join(methods)}.npy'), y_interp)
        np.save(osp.join(self.odir, f'y_pool_interpolate_{"_".join(methods)}.npy'), y_interp_pool)

        self.ofile.close()

    @staticmethod
    def modified_z_score(inp):
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        dev = inp - np.median(inp)
        median_abs_dev = np.median(np.abs(dev))
        return 0.6745 * dev / median_abs_dev

    def find_zeros_along_diagonal(self):
        '''Find rows/cols of y where y_ii = 0.'''
        diag = self.y.diagonal().copy()
        print(f'Zeros', file = self.ofile)
        for i, val in enumerate(diag):
            left = self.start + i * self.res
            right = self.start + (i+1) * self.res
            if val == 0:
                self.interp_locations.add(i)
                print(i, f'chr{self.chrom}:{left}-{right}', file = self.ofile)


    def find_sum_outliers(self, cutoff, modified = False):
        '''Find row_sums of y defined as outliers by modified zscore test.'''
        row_sums = np.sum(self.y, 0)
        if modified:
            zscores = Interpolater.modified_z_score(row_sums)
            if cutoff is None:
                cutoff = 5
        else:
            zscores = ss.zscore(row_sums, ddof=1)

        plt.plot(zscores)
        plt.axhline(-1 * cutoff, color = 'k')
        # plt.yscale('log')
        plt.xlabel('Polymer Distance (beads)', fontsize = 16)
        if modified:
            plt.ylabel('Modified ZScore', fontsize = 16)
            plt.savefig(osp.join(self.dir, 'sum_mzscore.png'))
        else:
            plt.ylabel('ZScore', fontsize = 16)
            plt.savefig(osp.join(self.dir, 'sum_zscore.png'))
        plt.close()

        print(f'Sum Outliers', file = self.ofile)
        for i, zscore in enumerate(zscores):
            left = self.start + i * self.res
            right = self.start + (i+1) * self.res
            if zscore < -1 * cutoff:
                print(i, f'chr{self.chrom}:{left}-{right}', zscore, file = self.ofile)
                self.interp_locations.add(i)
                if np.sum(self.y[i]) > np.median(row_sums):
                    print(i, 'gt median')

    def find_low_mappability(self, cutoff):
        '''Find rows/cols of y with low read mappability.'''
        m = len(self.y)
        if cutoff is None:
            cutoff = 0.6

        bw = pyBigWig.open('/home/erschultz/sequences_to_contact_maps/chip_seq_data/wgEncodeDukeMapabilityUniqueness35bp.bigWig')

        print('Low Mappability', file = self.ofile)
        mappability = []
        for i in range(m):
            left = self.start + i * self.res
            right = self.start + (i+1) * self.res
            stat = bw.stats(f'chr{self.chrom}', left, right)[0]
            mappability.append(stat)

            if stat < cutoff:
                print(i, f'chr{self.chrom}:{left}-{right}', stat, file = self.ofile)
                self.interp_locations.add(i)

        plt.plot(mappability)
        plt.axhline(cutoff, color = 'k')
        plt.xlabel('Polymer Distance (beads)', fontsize = 16)
        # plt.yscale('log')
        plt.ylabel('Mappability', fontsize = 16)
        plt.savefig(osp.join(self.odir, f'mappability-{cutoff}.png'))
        plt.close()

    def find_high_sparsity(self, cutoff):
        '''Find rows/cols of y with high sparsity.'''
        m = len(self.y)
        if cutoff is None:
            cutoff = 0.5

        sparsity = np.count_nonzero(self.y==0, axis=1) / m

        plt.plot(sparsity)
        plt.axhline(cutoff, color = 'k')
        plt.yscale('log')
        plt.xlabel('Polymer Distance (beads)', fontsize = 16)
        plt.ylabel('Sparsity', fontsize = 16)
        plt.savefig(osp.join(self.odir, 'sparsity.png'))
        plt.close()

        print('High Sparsity', file = self.ofile)
        for i, val in enumerate(sparsity):
            if val > cutoff:
                print(i, f'chr{self.chrom}:{left}-{right}', val, file = self.ofile)
                self.interp_locations.add(i)

    def find_sparsity_outliers(self, cutoff, modified=False):
        '''Find rows/cols of sparsity deemed an outlier.'''
        m = len(self.y)

        sparsity = np.count_nonzero(self.y==0, axis=1) / m
        if modified:
            zscores = Interpolater.modified_z_score(sparsity)
            if cutoff is None:
                cutoff = 5
        else:
            zscores = ss.zscore(sparsity, ddof=1)

        plt.plot(zscores)
        plt.axhline(cutoff, color = 'k')
        plt.xlabel('Polymer Distance (beads)', fontsize = 16)
        plt.yscale('log')
        if modified:
            plt.ylabel('Modified ZScore', fontsize = 16)
            plt.savefig(osp.join(self.odir, 'sparsity_mzscore.png'))
        else:
            plt.ylabel('ZScore', fontsize = 16)
            plt.savefig(osp.join(self.odir, 'sparsity_zscore.png'))
        plt.close()

        print('Sparsity Outliers', file = self.ofile)
        for i, val in enumerate(zscores):
            left = self.start + i * self.res
            right = self.start + (i+1) * self.res

            if val > cutoff:
                print(i, f'chr{self.chrom}:{left}-{right}', val, file = self.ofile)
                self.interp_locations.add(i)

    def linear_interpolate(self, fill_mean = False):
        '''
        Linearly interpolates each row in rows of contact map y.

        Entry y_ij will be linear interpolant of y_i-1j-1 and y_i+1j+1
        in order to better preserve diagonal information
        '''
        m = len(self.y)
        y = self.y.copy()
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(self.y)
        for i in self.interp_locations:
            for j in range(m):
                if fill_mean:
                    y[i,j] = meanDist[abs(i-j)]
                else:
                    # choose left boundary
                    left_i = i-1
                    left_j = j-1
                    while left_i in self.interp_locations or left_j in self.interp_locations:
                        left_i -= 1
                        left_j -= 1
                    # choose right boundary
                    right_i = i+1
                    right_j = j+1
                    while right_i in self.interp_locations or right_j in self.interp_locations:
                        right_i += 1
                        right_j += 1

                    left = None
                    if left_i >= 0 and left_j >= 0:
                        left = y[left_i, left_j]

                    right = None
                    if right_i < m and right_j < m:
                        right = y[right_i, right_j]

                    if left is None and right is None:
                        assert "this shouldn't happen"
                    elif left is None:
                        y[i,j] = right
                    elif right is None:
                        y[i,j] = left
                    else:
                        y[i,j] = np.interp(i, [left_i, right_i], [left, right])
                    # print(i,j, y[i,j], (left_i, left_j), left, (right_i, right_j), right)
                y[j,i] = y[i,j]

        return y


def main():
    range(101, 106)
    for sample in [2002]:
        # Interpolater('dataset_11_14_22', sample, ['zeros'])
        # Interpolater('dataset_11_14_22', sample, ['sum_mzscore'])
        # Interpolater('dataset_11_14_22', sample, ['sum_zscore-5'])
        # Interpolater('dataset_11_14_22', sample, ['sparsity'])
        # Interpolater('dataset_11_14_22', sample, ['sparsity_mzscore'])
        # Interpolater('dataset_11_14_22', sample, ['sparsity_zscore-5'])
        # Interpolater('dataset_11_14_22', sample, ['mappability-0.6'])
        # Interpolater('dataset_11_14_22', sample, ['mappability-0.5'])
        # Interpolater('dataset_11_14_22', sample, ['mappability-0.3'])
        # Interpolater('dataset_11_14_22', sample, ['sum_mzscore', 'mappability'])
        # Interpolater('dataset_11_14_22', sample, ['sparsity_zscore-5', 'mappability-0.5'])
        Interpolater('dataset_11_14_22', sample, ['zeros', 'mappability-0.5'])

if __name__ == '__main__':
    main()
