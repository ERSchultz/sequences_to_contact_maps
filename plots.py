import csv
import json
import math
import os
import os.path as osp
from shutil import rmtree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.argparse_utils import (argparse_setup, finalize_opt,
                                  get_base_parser, get_opt_header, opt2list)
from utils.plotting_utils import (plot_centroid_distance, plot_combined_models,
                                  plot_diag_chi, plot_sc_contact_maps,
                                  plot_xyz_gif, plotting_script)
from utils.utils import DiagonalPreprocessing
from utils.xyz_utils import xyz_load, xyz_write


def update_result_tables(model_type = None, mode = None, output_mode = 'contact'):
    if model_type is None:
        model_types = ['Akita', 'DeepC', 'UNet', 'GNNAutoencoder', 'GNNAutoencoder2',
                        'ContactGNN', 'ContactGNNEnergy']
        modes = [None, None, None, 'GNN', 'GNN']
        output_modes = ['contact', 'contact', 'contact', 'contact', 'sequence', 'energy']
    else:
        model_types = [model_type]
        modes = [mode]
        output_modes = [output_mode]
    for model_type, mode, output_mode in zip(model_types, modes, output_modes):
        # set up header row
        opt_list = get_opt_header(model_type, mode)
        if output_mode == 'contact':
            opt_list.extend(['Final Validation Loss', 'PCA Accuracy Mean',
                'PCA Accuracy Std', 'PCA Spearman Mean', 'PCA Spearman Std',
                'PCA Pearson Mean', 'PCA Pearson Std', 'Overall Pearson Mean',
                'Overall Pearson Std'])
        elif output_mode == 'sequence':
            opt_list.extend(['Final Validation Loss', 'AUC'])
        elif output_mode == 'energy':
            opt_list.extend(['Final Validation Loss'])
        else:
            raise Exception('Unknown output_mode {}'.format(output_mode))
        results = [opt_list]

        # get data
        model_path = osp.join('results', model_type)
        parser = get_base_parser()
        for id in range(1, 500):
            id_path = osp.join(model_path, str(id))
            if osp.isdir(id_path):
                txt_file = osp.join(id_path, 'argparse.txt')
                if osp.exists(txt_file):
                    opt = parser.parse_args(['@{}'.format(txt_file)])
                    opt.id = int(id)
                    opt = finalize_opt(opt, parser, local = True, debug = True)
                    opt_list = opt2list(opt)
                    if output_mode == 'contact':
                        with open(osp.join(id_path, 'PCA_results.txt'), 'r') as f:
                            f.readline()
                            acc = f.readline().split(':')[1].strip().split(' +- ')
                            spearman = f.readline().split(':')[1].strip().split(' +- ')
                            pearson = f.readline().split(':')[1].strip().split(' +- ')
                        with open(osp.join(id_path, 'out.log'), 'r') as f:
                            for line in f:
                                if line.startswith('Final val loss: '):
                                    final_val_loss = line.split(':')[1].strip()
                                elif line.startswith('Overall Pearson R: '):
                                    dist_pearson = line.split(':')[1].strip().split(' $\pm$ ')
                        opt_list.extend([final_val_loss, acc[0], acc[1], spearman[0],
                                        spearman[1], pearson[0], pearson[1],
                                        dist_pearson[0], dist_pearson[1]])
                    elif output_mode == 'sequence':
                        final_val_loss = None; auc = None
                        with open(osp.join(id_path, 'out.log'), 'r') as f:
                            for line in f:
                                if line.startswith('Final val loss: '):
                                    final_val_loss = line.split(':')[1].strip()
                                elif line.startswith('AUC: '):
                                    auc = line.split(':')[1].strip()
                        opt_list.extend([final_val_loss, auc])
                    elif output_mode == 'energy':
                        final_val_loss = None
                        with open(osp.join(id_path, 'out.log'), 'r') as f:
                            for line in f:
                                if line.startswith('Final val loss: '):
                                    final_val_loss = line.split(':')[1].strip()
                        opt_list.extend([final_val_loss])
                    results.append(opt_list)

        ofile = osp.join(model_path, 'results_table.csv')
        with open(ofile, 'w', newline = '') as f:
            wr = csv.writer(f)
            wr.writerows(results)

def plot_xyz_gif_wrapper():
    dir = '/home/erschultz/dataset_test/samples/sample1'
    file = osp.join(dir, 'data_out/output.xyz')

    m=200

    x = np.load(osp.join(dir, 'x.npy'))[:m, :]
    xyz = xyz_load(file, multiple_timesteps=True)[::50, :m, :]
    print(xyz.shape)
    xyz_write(xyz, osp.join(dir, 'data_out/output_x.xyz'), 'w', x = x)

    plot_xyz_gif(xyz, x, dir)

def plot_diag_vs_diag_chi():
    dir = '/home/erschultz/dataset_test_diag1024/samples'
    data = []
    ids = set()
    m_dict = {} # sample_id : m
    for file in os.listdir(dir):
        id = int(file[6:])

        ids.add(id)
        print(id)
        file_dir = osp.join(dir, file)
        y = np.load(osp.join(file_dir, 'y.npy')).astype(np.float64)
        y /= np.max(y)
        m = len(y)
        m_dict[id] = m
        with open(osp.join(file_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        chi_diag = config['diag_chis']
        k = len(chi_diag)

        diag_means = DiagonalPreprocessing.genomic_distance_statistics(y)
        max_diag_mean = np.max(diag_means[3:])
        # diag_means /= max_diag_mean

        temp = []
        prev_diag_chi = chi_diag[0]
        for i in range(1, m):
            diag_chi = chi_diag[math.floor(i/(m/k))]
            temp.append(diag_means[i])
            if diag_chi != prev_diag_chi:
                mean = np.mean(temp)
                data.append([mean, prev_diag_chi, id])
                temp = []
                prev_diag_chi = diag_chi
        else:
            mean = np.mean(temp)
            data.append([mean, diag_chi, id])


    data = np.array(data)
    cmap = matplotlib.cm.get_cmap('tab20')
    ls_arr = ['solid', 'dotted', 'dashed', 'dashdot']
    m_arr = []
    prev_m = None

    fig, ax = plt.subplots()
    for id in sorted(ids):
        m = m_dict[id]
        if m != prev_m:
            m_arr.append(m)
            ls = ls_arr[len(m_arr) - 1]
            prev_m = m
        where = np.equal(data[:, 2], id)
        ax.plot(data[where, 1], data[where, 0], color = cmap(id % cmap.N), ls = ls)
    ax.set_xlabel('Diagonal Parameter', fontsize=16)
    ax.set_ylabel('Contact Probability', fontsize=16)
    ax.set_yscale('log')


    for id in sorted(ids):
        ax.plot(np.NaN, np.NaN, color = cmap(id % cmap.N), label = id)

    ax2 = ax.twinx()
    for style, m in zip(ls_arr, m_arr):
        ax2.plot(np.NaN, np.NaN, ls = style, label = m, c = 'k')
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc = 1, title = 'sample')
    ax2.legend(loc = 3, title = 'm')
    plt.tight_layout()
    plt.savefig(osp.join(osp.split(dir)[0], 'chi_diag.png'))
    plt.close()

def plot_mean_vs_genomic_distance_comparison(dir, samples):
    # normalize_dict = {}
    # normalize_dict[512] = np.loadtxt(osp.join(dir, 'samples/sample203/meanDist.txt'))
    # normalize_dict[1024] = np.loadtxt(osp.join(dir, 'samples/sample204/meanDist.txt'))
    # normalize_dict[2048] = np.loadtxt(osp.join(dir, 'samples/sample205/meanDist.txt'))

    cmap = matplotlib.cm.get_cmap('tab10')

    # sort samples
    samples_arr = []
    m_arr = []
    for sample in samples:
        y = np.load(osp.join(dir, f'samples/sample{sample}', 'y.npy'))
        m = len(y)
        samples_arr.append(sample)
        m_arr.append(m)
    samples_sort = [samples_arr for _, samples_arr in sorted(zip(m_arr, samples_arr), reverse = True)]


    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    first = True
    for sample in samples_sort:
        y = np.load(osp.join(dir, f'samples/sample{sample}', 'y.npy'))
        m = len(y)
        print(sample, m)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                                zero_diag = False, zero_offset = 0)
        # meanDist = meanDist[:int(0.25*m)]

        meanDist *= (1/np.max(meanDist))
        ind = int(np.log2(m))
        ax.plot(np.arange(0, len(meanDist)), meanDist, label = m, color = cmap(ind % cmap.N))

        diag_chis_file = osp.join(dir, f'samples/sample{sample}', 'diag_chis.npy')
        if osp.exists(diag_chis_file):
            diag_chis = np.load(diag_chis_file)
            # diag_chis = diag_chis[:int(0.25*len(diag_chis))]
            ax2.plot(np.linspace(0, len(meanDist), len(diag_chis)), diag_chis, ls='--', color = cmap(ind % cmap.N))
            ax2.set_ylabel('Diagonal Parameter', fontsize = 16)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance', fontsize = 16)
    ax.legend(loc='upper right', title='Beads')
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'meanDist_{samples}.png'))
    plt.close()

    print('\n')

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    first = True
    for sample in samples_sort:
        y = np.load(osp.join(dir, f'samples/sample{sample}', 'y.npy'))
        m = len(y)
        print(sample, m)
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                                zero_diag = False, zero_offset = 0)
                                                # [:int(0.25*m)]
        meanDist *= (1/np.max(meanDist))
        ind = int(np.log2(m))
        ax.plot(np.linspace(0, 1, len(meanDist)), meanDist, label = m, color = cmap(ind % cmap.N))

        diag_chis_file = osp.join(dir, f'samples/sample{sample}', 'diag_chis.npy')
        if osp.exists(diag_chis_file):
            diag_chis = np.load(diag_chis_file)
            # diag_chis = diag_chis[:int(0.25*len(diag_chis))]
            diag_chis = diag_chis[:int(0.25*len(diag_chis))]
            ax2.plot(np.linspace(0, 1, len(diag_chis)), diag_chis, ls='--', color = cmap(ind % cmap.N))
            ax2.set_ylabel('Diagonal Parameter', fontsize = 16)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance (%)', fontsize = 16)
    ax.legend(loc='upper right', title='Beads')
    plt.tight_layout()
    plt.savefig(osp.join(dir, f'meanDist_{samples}_%.png'))
    plt.close()

    print('\n')


def main():
    opt = argparse_setup()
    print(opt, '\n')
    plotting_script(None, opt)
    # interogateParams(None, opt)

    # cleanup
    if opt.root is not None and opt.delete_root:
        rmtree(opt.root)

if __name__ == '__main__':
    # plot_diag_vs_diag_chi()
    # plot_xyz_gif_wrapper()
    # plot_centroid_distance(parallel = True, samples = [34, 35, 36])
    # update_result_tables('ContactGNNEnergy', None, 'energy')
    # plot_mean_vs_genomic_distance_comparison('/home/erschultz/dataset_test_diag', [200, 201, 202])
    # plot_mean_vs_genomic_distance_comparison('/home/erschultz/dataset_test_diag', [160, 161, 162, 163, 164])
    # plot_mean_vs_genomic_distance_comparison('/home/erschultz/sequences_to_contact_maps/dataset_07_20_22', [1, 2, 3, 4, 5, 6])
    # plot_combined_models('test', [154, 159])
    # main()
    plot_diag_chi(np.array([25.27974, 4.89618, 6.25198, 6.64107, 6.82722, 7.42828, 6.97835, 6.65819,
                7.48775, 7.14076, 6.99229, 7.06783, 6.76844, 7.06846, 6.90215, 6.92753,
                7.08992, 7.23810, 7.42200, 7.54987, 8.35464, 8.76678, 9.05908, 10.20939,
                10.51292, 11.51608, 12.68125, 13.95166, 16.26210, 15.83551, 16.99994, 18.53027]),
                1024, '', dense=True)
