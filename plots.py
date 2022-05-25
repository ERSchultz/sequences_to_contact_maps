import csv
import json
import math
import os
import os.path as osp
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
from utils.argparse_utils import (argparse_setup, finalize_opt,
                                  get_base_parser, get_opt_header, opt2list)
from utils.plotting_utils import (plot_centroid_distance, plot_combined_models,
                                  plot_sc_contact_maps, plot_xyz_gif,
                                  plotting_script)
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
    dir = '/home/erschultz/dataset_test3/samples'
    data = []
    ids = set()
    for file in os.listdir(dir):
        id = int(file[6:])
        ids.add(id)
        print(id)
        file_dir = osp.join(dir, file)
        y = np.load(osp.join(file_dir, 'y.npy')).astype(np.float64)
        y /= np.max(y)
        m = len(y)
        with open(osp.join(file_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        diag_chis = config['diag_chis']
        k = len(diag_chis)

        diag_means = DiagonalPreprocessing.genomic_distance_statistics(y)
        max_diag_mean = np.max(diag_means[3:])

        # for i in range(1, m):
        #     diag_chi = diag_chis[math.floor(i/(m/k))]
        #     mean = diag_means[i] / max_diag_mean
        #     data.append([mean, diag_chi, id])

        temp = []
        prev_diag_chi = diag_chis[0]
        for i in range(1, m):
            diag_chi = diag_chis[math.floor(i/(m/k))]
            temp.append(diag_means[i] )
            # / max_diag_mean
            if diag_chi != prev_diag_chi:
                mean = np.mean(temp)
                data.append([mean, prev_diag_chi, id])
                temp = []
                prev_diag_chi = diag_chi
        else:
            mean = np.mean(temp)
            data.append([mean, diag_chi, id])


    data = np.array(data)

    for id in ids:
        where = np.equal(data[:, 2], id)
        plt.plot(data[where, 1], data[where, 0], label = f'sample{id}')
    plt.xlabel('diag chi')
    plt.ylabel('mean')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    opt = argparse_setup()
    print(opt, '\n')
    plotting_script(None, opt)
    # interogateParams(None, opt)

    # cleanup
    if opt.root is not None and opt.delete_root:
        rmtree(opt.root)

if __name__ == '__main__':
    plot_diag_vs_diag_chi()
    # plot_xyz_gif_wrapper()
    # plot_centroid_distance(parallel = True, samples = [34, 35, 36])
    # update_result_tables('ContactGNNEnergy', 'GNN', 'energy')
    # plot_combined_models('ContactGNNEnergy', [150, 158])
    # main()
