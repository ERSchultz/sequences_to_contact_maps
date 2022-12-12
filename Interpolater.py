import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import scipy.stats as ss
from utils.plotting_utils import plot_matrix
from utils.utils import rescale_matrix


class Interpolater():
    def __init__(self, methods, dataset = None, sample = None, dir = None,
                chrom = None, start = None, end = None, res = None):
        '''
        Class for interpolating contact maps.

        Must either specify dataset and sample or specify dir.

        Inputs chrom, start, end, and res are required for method=mappability,
        otherwise are optionally used during writing to log.log

        Inputs:
            methods (list): list of methods to identify rows to interpolate
            dataset: dataset for input data
            sample: sample for input data
            dir: dir for input and output (must specify if dataset/sample = None)
            chrom: chromosome (enter as 5 not chr5)
            start: start in bp
            end: end in bp
            res: resolution in bp
        '''
        self.methods = methods
        if dataset is not None and sample is not None:
            self.dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        else:
            assert dir is not None, 'must specify dir or dataset and sample'
            self.dir = dir


        self.odir = osp.join(self.dir, 'Interpolation') # output directory
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)
        self.odir = osp.join(self.odir, "_".join(methods))
        if not osp.exists(self.odir):
            os.mkdir(self.odir, mode = 0o755)
        self.ofile = open(osp.join(self.odir, 'log.log'), 'a')

        # if import_log_file exists, infer chrom, start, end, res
        import_log_file = osp.join(self.dir, 'import.log')
        if osp.exists(import_log_file):
            with open(import_log_file, 'r') as f:
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

    def run(self, y = None):
        '''Interpolate contact map y using self.methods.'''
        if y is None:
            y_file = osp.join(self.dir, 'y.npy')
            assert osp.exists(y_file), f'{y_file} does not exist - must specify y'
            self.y = np.load(y_file)
        else:
            self.y = y

        crop_left = 500
        crop_right = 1200
        # plot_matrix(self.y[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, 'y_crop.png'), vmax = 'mean')
        y_pool = rescale_matrix(self.y, 5)
        # plot_matrix(y_pool[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, 'y_pool_crop.png'), vmax = 'mean')
        # plot_matrix(y_pool, osp.join(self.odir, 'y_pool.png'), vmax = 'mean')

        self.interp_locations = set()
        for method in self.methods:
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

        print(f'Interpolating {len(self.interp_locations)} rows')

        # plot_matrix(self.y, osp.join(self.odir, f'y_lines_{"_".join(self.methods)}.png'),
                    # vmax = 'mean', lines = self.interp_locations)
        # plot_matrix(y_interp, osp.join(self.odir, f'y_{"_".join(self.methods)}.png'), vmax = 'mean')
        # plot_matrix(y_interp[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, f'y_crop_{"_".join(self.methods)}.png'), vmax = 'mean')
        y_interp_pool = rescale_matrix(y_interp, 5)
        plot_matrix(y_interp_pool, osp.join(self.odir, f'y_pool_{"_".join(self.methods)}.png'), vmax = 'mean')
        # plot_matrix(y_interp_pool[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, f'y_pool_crop_{"_".join(self.methods)}.png'), vmax = 'mean')

        np.save(osp.join(self.odir, f'y_interpolate_{"_".join(self.methods)}.npy'), y_interp)
        np.save(osp.join(self.odir, f'y_pool_interpolate_{"_".join(self.methods)}.npy'), y_interp_pool)

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
        for i, val in enumerate(diag):
            if val == 0:
                self.interp_locations.add(i)

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

        for i, zscore in enumerate(zscores):
            if zscore < -1 * cutoff:
                self.interp_locations.add(i)

    def find_low_mappability(self, cutoff):
        '''Find rows/cols of y with low read mappability.'''
        m = len(self.y)
        if cutoff is None:
            cutoff = 0.6

        # http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/
        bw = pyBigWig.open('/home/erschultz/sequences_to_contact_maps/chip_seq_data/wgEncodeDukeMapabilityUniqueness35bp.bigWig')

        mappability = []
        for i in range(m):
            left = self.start + i * self.res
            right = self.start + (i+1) * self.res
            stat = bw.stats(f'chr{self.chrom}', left, right)[0]
            mappability.append(stat)
            if stat < cutoff:
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

        for i, val in enumerate(sparsity):
            if val > cutoff:
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

        for i, val in enumerate(zscores):
            if val > cutoff:
                self.interp_locations.add(i)

    def linear_interpolate(self, fill_mean = False):
        '''
        Linearly interpolates each row in rows of contact map y.

        Entry y_ij will be linear interpolant of y_i-1j-1 and y_i+1j+1
        in order to better preserve diagonal information
        '''
        m = len(self.y)
        y = self.y.copy()
        for i in self.interp_locations:
            if self.chrom is not None and self.res is not None:
                left = self.start + i * self.res
                right = self.start + (i+1) * self.res
                print(i, f'chr{self.chrom}:{left}-{right}', file = self.ofile)
            else:
                print(i, file = self.ofile)

            for j in range(m):
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
    dataset = 'dataset_07_20_22'
    for sample in range(2010, 2011):
        # this is the recommended option
        interpolater = Interpolater(['zeros', 'mappability-0.7'], dataset, sample)
        interpolater.run()

        dir = f'/home/erschultz/{dataset}/samples/sample{sample+200}'
        if not osp.exists(dir):
            os.mkdir(dir, mode=0o755)
        shutil.copyfile(osp.join(interpolater.odir, 'y_pool_zeros_mappability-0.7.png'),
                        osp.join(dir, 'y.png'))
        shutil.copyfile(osp.join(interpolater.odir, 'y_pool_interpolate_zeros_mappability-0.7.npy'),
                        osp.join(dir, 'y.npy'))

        with open(osp.join(dir, 'import.log'), 'w') as f:
            f.write(f'Interpolation of sample {sample} with mappability-0.7 and zeros\n')
            f.write(f'chrom={interpolater.chrom}\n')
            f.write(f'start={interpolater.start}\n')
            f.write(f'end={interpolater.end}\n')
            f.write(f'resolution={interpolater.res * 5}\n')
            f.write('norm=NONE')


if __name__ == '__main__':
    main()
