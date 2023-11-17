import multiprocessing
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import scipy.stats as ss
import seaborn as sns
from liftover import get_lifter
from scripts.load_utils import load_import_log
# from pyliftover import LiftOver as get_lifter
from scripts.plotting_utils import RED_CMAP, plot_matrix
from scripts.utils import rescale_matrix


class Interpolater():
    def __init__(self, methods, dataset = None, sample = None, dir = None):
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

        self.genome = 'hg19' # default
        load_import_log(self.dir, self)

    def run(self, y = None, factor=5):
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
        y_pool = rescale_matrix(self.y, factor)
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

        print(f'Interpolating {len(self.interp_locations)} rows', file = self.ofile)

        # plot_matrix(self.y, osp.join(self.odir, f'y_lines_{"_".join(self.methods)}.png'),
                    # vmax = 'mean', lines = self.interp_locations)
        # plot_matrix(y_interp, osp.join(self.odir, f'y_{"_".join(self.methods)}.png'), vmax = 'mean')
        # plot_matrix(y_interp[crop_left:crop_right, crop_left:crop_right],
                    # osp.join(self.odir, f'y_crop_{"_".join(self.methods)}.png'), vmax = 'mean')
        y_interp_pool = rescale_matrix(y_interp, factor)
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

        if self.genome == 'hg38':
            converter = get_lifter('hg38', 'hg19')


        # http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/
        file = '/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/home/erschultz/chip_seq_data/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
        bw = pyBigWig.open(file)

        mappability = []
        for i in range(m):
            left = self.start + i * self.resolution
            right = self.start + (i+1) * self.resolution
            if self.genome == 'hg38':
                orig = f'{left}-{right}'
                left_result = converter.convert_coordinate(f'chr{self.chrom}', left)
                count = 0
                while len(left_result) == 0 and count < self.resolution:
                    # If left does not map to hg19, there will be no result
                    # this finds nearest result
                    left -= 1
                    count += 1
                    left_result = converter.convert_coordinate(f'chr{self.chrom}', left)

                right_result = converter.convert_coordinate(f'chr{self.chrom}', right)
                count = 0
                while len(right_result) == 0 and count < self.resolution:
                    right += 1
                    count += 1
                    right_result = converter.convert_coordinate(f'chr{self.chrom}', right)

                if len(left_result) == 0 or len(right_result) == 0:
                    # liftover failed
                    print(f'Warning liftover failed for {orig}')
                    mappability.append(np.NaN)
                    continue

                left = left_result[0][1]
                right = right_result[0][1]

            try:
                stat = bw.stats(f'chr{self.chrom}', left, right)[0]
            except RuntimeError:
                print(f'RuntimeError chr{self.chrom}', left, right)
                raise
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
            if self.chrom is not None and self.resolution is not None:
                left = self.start + i * self.resolution
                right = self.start + (i+1) * self.resolution
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

def wrapper(dataset, sample, factor):
    high_res_dir = f'/home/erschultz/{dataset}/samples_10k/sample{sample}'
    if not osp.exists(high_res_dir):
        print(f'{high_res_dir} does not exist')
        return

    # this is the recommended option
    interpolater = Interpolater(['zeros', 'mappability-0.7'], dir = high_res_dir)
    interpolater.run(factor = factor)

    odir = f'/home/erschultz/{dataset}/samples/'
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    odir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    assert odir != high_res_dir # prevent overwriting data on accident
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    shutil.copyfile(osp.join(interpolater.odir, 'y_pool_zeros_mappability-0.7.png'),
                    osp.join(odir, 'y.png'))
    shutil.copyfile(osp.join(interpolater.odir, 'y_pool_interpolate_zeros_mappability-0.7.npy'),
                    osp.join(odir, 'y.npy'))

    with open(osp.join(odir, 'import.log'), 'w') as f:
        f.write(f'Interpolation of sample {sample} with mappability-0.7 and zeros\n')
        f.write(f'{interpolater.url}\n')
        f.write(f'chrom={interpolater.chrom}\n')
        f.write(f'start={interpolater.start}\n')
        f.write(f'end={interpolater.end}\n')
        f.write(f'resolution={interpolater.resolution * factor}\n')
        f.write(f'norm={interpolater.norm}\n')
        f.write(f'genome={interpolater.genome}')

def main():
    dataset = 'dataset_HCT116_RAD21_KO'
    mapping = [(dataset, i, 5) for i in range(1, 9)]
    # serial version
    for dataset, i, factor in mapping:
        wrapper(dataset, i, factor)
    # with multiprocessing.Pool(2) as p:
        # p.starmap(wrapper, mapping)

def example_figure(dataset, sample):
    dir = osp.join(f'/home/erschultz/{dataset}')
    dir_raw = osp.join(dir, f'samples_10k/sample{sample}')
    y_raw = np.load(osp.join(dir_raw, 'y.npy'))
    # dir_interp = osp.join(dir, f'samples/sample{sample+200}')
    # y_interp = np.load(osp.join(dir_interp, 'y.npy'))
    y_interp = np.load(osp.join(dir_raw, 'Interpolation/zeros_mappability-0.7/y_interpolate_zeros_mappability-0.7.npy'))

    # get ticks in mb
    result = load_import_log(dir_raw)
    start = result['start_mb']
    end = result['end_mb']
    resoluton = result['resolutoin_mb']
    print(start, end, resolution)
    scale = 700
    all_ticks = np.arange(0, 2560, scale)
    all_ticks = np.append(all_ticks, 2560)
    print(all_ticks, len(all_ticks))

    all_tick_labels = np.arange(start, end, resolution*scale)
    all_tick_labels = np.append(all_tick_labels, end)
    print(all_tick_labels, len(all_tick_labels))
    all_tick_labels = [f'{i} Mb' for i in all_tick_labels]
    # return

    fig, (ax1, ax2, axcb) = plt.subplots(1, 3,
                                    gridspec_kw={'width_ratios':[1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2)
    vmin = 0
    vmax = np.mean(y_raw)

    s1 = sns.heatmap(y_raw, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                    ax = ax1, cbar = False)
    s1.set_title('Raw Hi-C', fontsize = 16)
    s1.set_yticks(all_ticks, labels = all_tick_labels, rotation='horizontal')
    s1.set_xticks(all_ticks, labels = all_tick_labels, rotation='horizontal')
    s2 = sns.heatmap(y_interp, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                    ax = ax2, cbar_ax = axcb)
    s2.set_title('Interpolation', fontsize = 16)
    s2.set_xticks(all_ticks, labels = all_tick_labels, rotation='horizontal')
    s2.set_yticks([])
    # axcb.yaxis.set_ticks_position('left')

    # fig.suptitle(f'Sample {sample}')
    plt.tight_layout()
    plt.savefig(osp.join(dir_raw, 'interp_figure.png'))
    plt.close()

def check_interpolation():
    '''Check if any of the experimental contact maps had too many rows/cols to be interpolated'''
    dir = '/home/erschultz/dataset_04_10_23'
    for sample in range(1, 28):
        s_dir = osp.join(dir, 'samples', f'sample{sample}')
        if not osp.exists(s_dir):
            continue
        try:
            with open(osp.join(s_dir, 'Interpolation/zeros_mappability-0.7/log.log')) as f:
                lines = f.readlines()
                last = lines[-1]
                rows = last.split(' ')[1]
                rows = int(rows)
                if rows > 1000:
                    print(sample, rows)
                    interp_dir = osp.join(dir, 'samples', f'sample{1000+sample}')
                    # if osp.exists(interp_dir):
                    #     shutil.rmtree(interp_dir)
                    # if osp.exists(s_dir):
                    #     shutil.rmtree(s_dir)
        except Exception as e:
            print(s_dir)
            raise e

def test_liftover():
    converter = get_lifter('hg38', 'hg19')
    i = 16140001
    result = converter.convert_coordinate(f'chr2', i)
    while len(result) == 0:
        i += 10
        result = converter.convert_coordinate(f'chr2', i)

    print(i, result)


if __name__ == '__main__':
    main()
    # test_liftover()
    # example_figure('dataset_02_04_23', 1)
    # check_interpolation()
