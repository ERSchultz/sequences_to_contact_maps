import csv
import json
import math
import os
import os.path as osp
import string
import sys
from collections import defaultdict
from shutil import rmtree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from pylib.utils import epilib
from pylib.utils.energy_utils import (calculate_all_energy, calculate_D,
                                      calculate_diag_chi_step, calculate_L,
                                      calculate_S)
from pylib.utils.plotting_utils import *
from pylib.utils.utils import load_json
from result_summary_plots import predict_chi_in_psi_basis
from scipy.ndimage import uniform_filter
from scripts.argparse_utils import (finalize_opt, get_base_parser,
                                    get_opt_header, opt2list)
from scripts.load_utils import (get_final_max_ent_folder, load_contact_map,
                                load_import_log, load_L)
from scripts.plotting_utils import (BLUE_RED_CMAP, RED_CMAP,
                                    plot_centroid_distance,
                                    plot_combined_models, plot_diag_chi,
                                    plot_sc_contact_maps, plot_xyz_gif,
                                    plotting_script)
from scripts.utils import DiagonalPreprocessing, pearson_round
from scripts.xyz_utils import xyz_load, xyz_write
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from makeLatexTable_new import *

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
        opt_list.pop(0) # pop model type
        if output_mode == 'contact':
            opt_list.extend(['Final Validation Loss', 'PCA Accuracy Mean',
                'PCA Accuracy Std', 'PCA Spearman Mean', 'PCA Spearman Std',
                'PCA Pearson Mean', 'PCA Pearson Std', 'Overall Pearson Mean',
                'Overall Pearson Std'])
        elif output_mode == 'sequence':
            opt_list.extend(['Final Validation Loss', 'AUC'])
        elif output_mode == 'energy':
            opt_list.extend(['Final Validation Loss', 'Regular Loss', 'Downsampling Loss', 'Upsampling Loss'])
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
                    try:
                        opt = parser.parse_args(['@{}'.format(txt_file)])
                    except:
                        print(id)
                        raise
                    opt.id = int(id)
                    opt = finalize_opt(opt, parser, local = True, debug = True)
                    opt_list = opt2list(opt)
                    opt_list.pop(0) # pop model_type
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
                                    final_val_loss = np.round(float(line.split(':')[1].strip()), 3)
                        r_loss = None; d_loss = None; u_loss = None;
                        if osp.exists(osp.join(id_path, 'loss_analysis.json')):
                            with open(osp.join(id_path, 'loss_analysis.json')) as f:
                                loss_dict = json.load(f)
                            if 'regular' in loss_dict.keys():
                                r_loss = loss_dict['regular']
                            if 'downsample' in loss_dict.keys():
                                d_loss = loss_dict['downsample']
                            if 'upsample' in loss_dict.keys():
                                u_loss = loss_dict['upsample']
                        opt_list.extend([final_val_loss, r_loss, d_loss, u_loss])
                    results.append(opt_list)

        ofile = osp.join(model_path, 'results_table.csv')
        with open(ofile, 'w', newline = '') as f:
            wr = csv.writer(f)
            wr.writerows(results)

def plot_xyz_gif_wrapper():
    dir = '/home/erschultz/dataset_02_04_23/samples/sample202/optimize_grid_b_140_phi_0.03-GNN403'
    file = osp.join(dir, 'production_out/output.xyz')
    m=256
    k=2

    # y = np.load(osp.join(dir, 'y.npy'))
    # y_diag = epilib.get_oe(y)
    # seqs = epilib.get_pcs(y_diag, k, normalize = True)
    # kmeans = KMeans(n_clusters = k)
    # kmeans.fit(y_diag)
    # x = np.zeros((m, k))
    # x[np.arange(m), kmeans.labels_] = 1

    # x = np.load(osp.join(dir, 'x.npy'))[:m, :]

    x = np.arange(1, m+1) # use this to color by m
    xyz = xyz_load(file, multiple_timesteps=True)[::, :m, :]
    print(xyz.shape)
    # xyz_write(xyz, osp.join(dir, 'production_out/output_x.xyz'), 'w', x = x)

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

def plot_mean_vs_genomic_distance_comparison(dir, samples = None, percent = False,
                                        ref_file = None, norm = False, logx = True,
                                        ln_transform = False,
                                        label = 'sample', params = True,
                                        zero_diag = False, zero_diag_offset = 1,
                                        skip_samples = []):
    cmap = matplotlib.cm.get_cmap('tab10')

    if samples is None:
        samples = [f[6:] for f in os.listdir(osp.join(dir, 'samples')) if f.startswith('sample')]
        samples = [f for f in samples if f not in skip_samples]
        ofile = osp.join(dir, f'meanDist_{label}.png')
    else:
        ofile = osp.join(dir, f'meanDist_{samples}_{label}.png')

    # sort samples
    samples_arr = []
    for sample in samples:
        file = osp.join(dir, f'samples/sample{sample}', 'y.npy')
        if osp.exists(file):
            y = np.load(file)
            samples_arr.append(sample)
    samples_sort = [samples_arr for samples_arr in sorted(samples_arr, reverse = True)]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    # process ref_file
    if ref_file is not None and osp.exists(ref_file):
        y = load_contact_map(ref_file, chrom = 15)
        meanDist_ref = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                        zero_diag = zero_diag,
                                        zero_offset = zero_diag_offset)
        meanDist_ref[50:] = uniform_filter(meanDist_ref[50:], 3, mode = 'constant')
        if norm:
            meanDist_ref /= np.max(meanDist_ref)
        if ln_transform:
            meanDist_ref = np.log(meanDist_ref+1e-8)
        ax.plot(meanDist_ref, label = 'ref', color = 'k')

    for i, sample in enumerate(samples_sort):
        sample_dir = osp.join(dir, f'samples/sample{sample}')
        y = np.load(osp.join(sample_dir, 'y.npy'))
        m = len(y)
        print(f'sample {sample}, m {m}')

        config_file = osp.join(sample_dir, 'config.json')
        config = None
        if osp.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                phi = config['phi_chromatin']
                if 'bond_length' in config:
                    bond_length = config['bond_length']

        # get param args
        method = None
        slope = None
        scale = None
        constant = None
        file = osp.join(dir, f'samples/sample{sample}', 'params.log')
        if osp.exists(file):
            with open(file, 'r') as f:
                line = f.readline()
                while line != '':
                    line = f.readline()
                    if line.startswith('Diag chi args:'):
                        line = f.readline().split(', ')
                        for arg in line:
                            if arg.startswith('diag_chi_method'):
                                method = arg.split('=')[1]
                            elif arg.startswith('diag_chi_slope'):
                                slope = float(arg.split('=')[1])
                            elif arg.startswith('diag_chi_scale'):
                                scale = arg.split('=')[1]
                                if scale == 'None':
                                    scale = None
                                else:
                                    scale = float(scale)
                            elif arg.startswith('diag_chi_constant'):
                                constant = float(arg.split('=')[1])

        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                        zero_diag = zero_diag,
                                        zero_offset = zero_diag_offset)
        # meanDist[50:] = uniform_filter(meanDist[50:], 3, mode = 'constant')
        if ln_transform:
            meanDist = np.log(meanDist+1e-8)

        if norm:
            meanDist /= np.max(meanDist)
        # print(meanDist)
        ind = None

        if label == 'm':
            fig_label = m
            ind = int(np.log2(m))
        elif label == 'chi':
            fig_label = None
            ind = i
        elif label == 'bond_length':
            fig_label = bond_length
            ind = i
        elif label == 'phi_chromatin':
            fig_label = phi
            ind = i
        elif label == 'sample':
            fig_label = sample
        elif label == 'scale':
            fig_label = scale
            ind = int(scale)
        elif label == 'slope':
            fig_label = slope
            ind = int(np.log2(slope))
        elif label == 'constant':
            fig_label = constant
            ind = int((constant + 15) / 5)
            print(ind)
        else:
            fig_label = None
        if ind is not None:
            ax.plot(np.arange(0, len(meanDist)), meanDist, label = fig_label, color = cmap(ind % cmap.N))
        else:
            ax.plot(np.arange(0, len(meanDist)), meanDist, label = fig_label)


        if params and config is not None:
            # find chi transition points
            diag_chi_step = calculate_diag_chi_step(config)

            if ind is not None:
                ax2.plot(diag_chi_step, ls='--', color = cmap(ind % cmap.N))
            else:
                ax2.plot(diag_chi_step, ls='--')
            ax2.set_ylabel('Diagonal Parameter', fontsize = 16)



    if not ln_transform:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(None, ymax + 10)
    ax.set_ylim(10**-6, None)
    ax.set_ylabel('Contact Probability', fontsize = 16)
    ax.set_xlabel('Polymer Distance', fontsize = 16)
    ax.legend(loc='upper right', title=label)
    plt.title(method)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

    print('\n')

    if percent:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        for sample in samples_sort:
            y = np.load(osp.join(dir, f'samples/sample{sample}', 'y.npy'))
            m = len(y)
            print(sample, m)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob',
                                                    zero_diag = False, zero_offset = 0)
                                                    # [:int(0.25*m)]
            if norm:
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
        if logx:
            ax.set_xscale('log')
        ax.set_ylabel('Contact Probability', fontsize = 16)
        ax.set_xlabel('Polymer Distance (%)', fontsize = 16)
        ax.legend(loc='upper right', title='Beads')
        plt.title(method)
        plt.tight_layout()
        plt.savefig(osp.join(dir, f'meanDist_{samples}_%.png'))
        plt.close()

        print('\n')

def main(id):
    model_type = 'ContactGNNEnergy'
    argparse_path = osp.join('/home/erschultz/sequences_to_contact_maps/results', model_type, f'{id}/argparse.txt')
    parser = get_base_parser()
    sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
    opt = parser.parse_args(['@{}'.format(argparse_path)])
    opt.id = id
    print(opt)
    opt = finalize_opt(opt, parser, local = True, debug = True)
    if isinstance(opt.data_folder, list):
        assert len(opt.data_folder) == 1
        opt.data_folder = opt.data_folder[0]
    data_folder_split = osp.normpath(opt.data_folder).split(os.sep)
    opt.data_folder = osp.join('/home/erschultz',data_folder_split[-1]) # use local dataset
    print(opt.data_folder)
    opt.device = torch.device('cpu')
    opt.plot_predictions = False
    print(opt)
    plotting_script(None, opt, samples = [410, 653, 1462, 1801, 2290])
    # interogateParams(None, opt)

def plot_GNN_vs_PCA(dataset, k, GNN_ID):
    dir = f'/home/erschultz/{dataset}/samples'
    for sample in range(1001, 1010):
        sample_dir = osp.join(dir, f'sample{sample}')
        if not osp.exists(sample_dir):
            print(f'sample_dir does not exist: {sample_dir}')
            continue

        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        y_diag = DiagonalPreprocessing.process(y, meanDist)

        grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')
        max_ent_dir = f'{grid_dir}-max_ent{k}'
        gnn_dir = f'{grid_dir}-GNN{GNN_ID}'

        final = get_final_max_ent_folder(max_ent_dir)
        with open(osp.join(final, 'distance_pearson.json'), 'r') as f:
            # TODO this is after final iteration, not convergence
            pca_results = json.load(f)
        y_pca = np.load(osp.join(final, 'y.npy')).astype(np.float64)
        y_pca /= np.mean(np.diagonal(y_pca))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_pca)
        y_pca_diag = DiagonalPreprocessing.process(y_pca, meanDist)
        L = load_L(max_ent_dir)
        with open(osp.join(final, 'config.json'), 'r') as f:
            config = json.load(f)
        diag_chis_continuous = calculate_diag_chi_step(config)
        D = calculate_D(diag_chis_continuous)
        S = calculate_S(L, D)


        with open(osp.join(gnn_dir, 'distance_pearson.json'), 'r') as f:
            gnn_results = json.load(f)
        y_gnn = np.load(osp.join(gnn_dir, 'y.npy')).astype(np.float64)
        y_gnn /= np.mean(np.diagonal(y_gnn))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_gnn)
        y_gnn_diag = DiagonalPreprocessing.process(y_gnn, meanDist)

        S_gnn = np.load(osp.join(gnn_dir, 'S.npy'))

        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0, 'blue'),
                                                 (0.5, 'white'),
                                                  (1, 'red')], N=126)

        fig, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                        gridspec_kw={'width_ratios':[1,1,1,0.08]})
        fig.set_figheight(6)
        fig.set_figwidth(6*2.5)
        fig.suptitle(f'Sample {sample}', fontsize = 16)
        vmin = 0
        vmax = np.mean(y)
        s1 = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                        ax = ax1, cbar = False)
        s1.set_title(r'Experimental Contact Map, $H$', fontsize = 16)
        s2 = sns.heatmap(y_pca, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                        ax = ax2, cbar = False)
        title = (r'Max Ent $\hat{H}$'
                f' (k={k})'
                f'\nSCC={np.round(pca_results["scc_var"], 3)}')
        s2.set_title(title, fontsize = 16)
        s2.set_yticks([])
        s3 = sns.heatmap(y_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                        ax = ax3, cbar_ax = axcb)
        title = (f'GNN-{GNN_ID} '
                r'$\hat{H}$'
                f'\nSCC={np.round(gnn_results["scc_var"], 3)}')
        s3.set_title(title, fontsize = 16)
        s3.set_yticks([])

        # s1.set_xticks([])
        # s1.set_yticks([])
        # s2.set_xticks([])


        plt.tight_layout()
        plt.savefig(osp.join(sample_dir, 'PCA_vs_GNN.png'))
        plt.close()

        fig, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                        gridspec_kw={'width_ratios':[1,1,1,0.08]})
        fig.set_figheight(6)
        fig.set_figwidth(6*2.5)
        fig.suptitle(f'Sample {sample}', fontsize = 16)
        vmin = np.nanpercentile(y_diag, 1)
        vmax = np.nanpercentile(y_diag, 99)
        d_vmin = 1-vmin
        d_vmax = vmax-1
        d = max(d_vmax, d_vmin)
        vmin = 1 - d
        vmax = 1 + d
        s1 = sns.heatmap(y_diag, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax1, cbar = False)
        s1.set_title(r'Experimental Contact Map, $H^{diag}$', fontsize = 16)
        s2 = sns.heatmap(y_pca_diag, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax2, cbar = False)
        title = (r'Max Ent $\hat{H}^{diag}$'
                f' (k={k})'
                f'\nSCC={np.round(pca_results["scc_var"], 3)}')
        s2.set_title(title, fontsize = 16)
        s2.set_yticks([])
        s3 = sns.heatmap(y_gnn_diag, linewidth = 0, vmin = vmin, vmax = vmax, cmap = BLUE_RED_CMAP,
                        ax = ax3, cbar_ax = axcb)
        title = (f'GNN-{GNN_ID} '
                r'$\hat{H}^{diag}$'
                f'\nSCC={np.round(gnn_results["scc_var"], 3)}')
        s3.set_title(title, fontsize = 16)
        s3.set_yticks([])

        plt.tight_layout()
        plt.savefig(osp.join(sample_dir, 'PCA_vs_GNN_diag.png'))
        plt.close()


        fig, (ax1, ax2, ax3, axcb) = plt.subplots(1, 4,
                                        gridspec_kw={'width_ratios':[1,1,1,0.08]})
        fig.set_figheight(6)
        fig.set_figwidth(6*2.5)
        fig.suptitle(f'Sample {sample}', fontsize = 16)
        arr = np.array([S, S_gnn])
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
        s1 = sns.heatmap(S, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap2,
                        ax = ax1, cbar = False)
        s1.set_title(f'Max Ent (k={k}) '+ r'$\hat{S}$', fontsize = 16)
        s1.set_yticks([])
        s2 = sns.heatmap(S_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap2,
                        ax = ax2, cbar = False)
        s2.set_title(f'GNN-{GNN_ID} '+r'$\hat{S}$', fontsize = 16)
        s2.set_yticks([])
        s3 = sns.heatmap(S - S_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap2,
                        ax = ax3, cbar_ax = axcb)
        title = ('Difference\n'
                r'(Max Ent $\hat{S}$ - GNN $\hat{S}$)')
        s3.set_title(title, fontsize = 16)
        s3.set_yticks([])

        # s1.set_xticks([])
        # s1.set_yticks([])
        # s2.set_xticks([])

        plt.tight_layout()
        plt.savefig(osp.join(sample_dir, 'PCA_vs_GNN_S.png'))
        plt.close()

        # compare P(s)
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
        meanDist_pca = DiagonalPreprocessing.genomic_distance_statistics(y_pca, 'prob')
        meanDist_gnn = DiagonalPreprocessing.genomic_distance_statistics(y_gnn, 'prob')
        for arr, fig_label, c in zip([meanDist, meanDist_pca, meanDist_gnn], ['Experiment', 'Max Ent', f'GNN-{GNN_ID}'], ['k', 'b', 'r']):
            ax.plot(np.arange(0, len(arr)), arr, label = fig_label, color = c)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ymin, ymax = ax2.get_ylim()
        # ax2.set_ylim(None, ymax + 10)
        # ax.set_ylim(10**-6, None)
        ax.set_ylabel('Contact Probability, P(s)', fontsize = 16)
        ax.set_xlabel('Polymer Distance, s', fontsize = 16)
        ax.legend(loc='upper right')
        plt.tight_layout()

        plt.savefig(osp.join(sample_dir, 'PCA_vs_GNN_p_s.png'))
        plt.close()

def plot_first_PC(dataset, k, GNN_ID):
    dir = f'/home/erschultz/{dataset}/samples'
    samples = list(range(201, 221))

    # collect data
    data = defaultdict(lambda: dict()) # sample : method : pcs
    for sample in samples:
        sample_dir = osp.join(dir, f'sample{sample}')
        assert osp.exists(sample_dir), f'sample_dir does not exist: {sample_dir}'

        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))

        grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')

        max_ent_dir = f'{grid_dir}-max_ent{k}'
        final = get_final_max_ent_folder(max_ent_dir)
        with open(osp.join(final, 'distance_pearson.json'), 'r') as f:
            # TODO this is after final iteration, not convergence
            pca_results = json.load(f)
        y_pca = np.load(osp.join(final, 'y.npy')).astype(np.float64)
        y_pca /= np.mean(np.diagonal(y_pca))

        gnn_dir = f'{grid_dir}-GNN{GNN_ID}'
        with open(osp.join(gnn_dir, 'distance_pearson.json'), 'r') as f:
            gnn_results = json.load(f)
        y_gnn = np.load(osp.join(gnn_dir, 'y.npy')).astype(np.float64)
        y_gnn /= np.mean(np.diagonal(y_gnn))


        # compare PCs
        pcs = epilib.get_pcs(epilib.get_oe(y), 12, align = True).T
        pcs_pca = epilib.get_pcs(epilib.get_oe(y_pca), 12, align = True).T
        pcs_gnn = epilib.get_pcs(epilib.get_oe(y_gnn), 12, align = True).T

        # change sign of first pc s.t corr is positive
        pcs_pca[0] *= np.sign(pearson_round(pcs[0], pcs_pca[0]))
        pcs_gnn[0] *= np.sign(pearson_round(pcs[0], pcs_gnn[0]))
        data[sample]['max_ent'] = pcs_pca
        data[sample]['gnn'] = pcs_gnn
        data[sample]['experiment'] = pcs


    # plot first few 2 pcs per sample
    for sample in samples:
        sample_dir = osp.join(dir, f'sample{sample}')
        pcs_pca = data[sample]['max_ent']
        pcs_gnn = data[sample]['gnn']
        pcs = data[sample]['experiment']

        rows = 2; cols = 1
        row = 0; col = 0
        fig, axes = plt.subplots(rows, cols)
        fig.set_figheight(12)
        fig.set_figwidth(16)
        for i in range(rows*cols):
            ax = axes[row]
            ax.plot(pcs[i], label = 'Experiment', color = 'k')
            ax.plot(pcs_pca[i], label = 'Max Ent', color = 'b')
            ax.plot(pcs_gnn[i], label = f'GNN-{GNN_ID}', color = 'r')
            ax.set_title(f'PC {i+1}\nCorr(Exp, GNN)={pearson_round(pcs[i], pcs_gnn[i])}')
            ax.legend()

            col += 1
            if col > cols-1:
                col = 0
                row += 1
        plt.savefig(osp.join(sample_dir, 'pc_comparison.png'))

    # plot first pc for all samples
    rows = 3; cols = 4
    row = 0; col = 0
    fig, axes = plt.subplots(rows, cols)
    fig.set_figheight(12)
    fig.set_figwidth(16)
    for i in range(rows*cols):
        ax = axes[row, col]
        sample = samples[i]
        pcs_pca = data[sample]['max_ent']
        pcs_gnn = data[sample]['gnn']
        pcs = data[sample]['experiment']

        ax.plot(pcs[0], label = 'Experiment', color = 'k')
        ax.plot(pcs_pca[0], label = 'Max Ent', color = 'b')
        ax.plot(pcs_gnn[0], label = f'GNN-{GNN_ID}', color = 'r')
        ax.set_title(f'Sample {sample}\nCorr(Exp, GNN)={pearson_round(pcs[0], pcs_gnn[0])}')
        ax.legend()

        col += 1
        if col > cols-1:
            col = 0
            row += 1
    plt.tight_layout()
    plt.savefig(osp.join(f'/home/erschultz/{dataset}/pc_comparison.png'))


def plot_Exp_vs_PCA(dataset, k=None):
    dir = f'/home/erschultz/{dataset}/samples'
    for sample in range(221, 222):
        sample_dir = osp.join(dir, f'sample{sample}')

        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))

        if k is None:
            pca_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03-max_ent')
        else:
            pca_dir = osp.join(sample_dir, f'PCA-normalize-E/k{k}/replicate1')

        with open(osp.join(pca_dir, 'distance_pearson.json'), 'r') as f:
            # TODO this is after final iteration, not convergence
            pca_results = json.load(f)
        y_pca = np.load(osp.join(pca_dir, 'y.npy')).astype(np.float64)
        y_pca /= np.mean(np.diagonal(y_pca))

        L = np.load(osp.join(pca_dir, 'L.npy'))
        max_it = get_final_max_ent_folder(pca_dir)
        with open(osp.join(max_it, 'config.json'), 'r') as f:
            config = json.load(f)
        diag_chis_continuous = calculate_diag_chi_step(config)
        D = calculate_D(diag_chis_continuous)
        S = calculate_S(L, D)

        plot_matrix(L, osp.join(pca_dir, 'L.png'), cmap='blue-red')
        plot_matrix(D, osp.join(pca_dir, 'D.png'))
        plot_matrix(S, osp.join(pca_dir, 'S.png'), cmap='blue-red')




        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0,    'white'),
                                                  (1,    'red')], N=126)
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                 [(0, 'blue'),
                                                 (0.5, 'white'),
                                                  (1, 'red')], N=126)

        fig, (axcb12, ax1, ax2, ax3, axcb3) = plt.subplots(1, 5,
                                        gridspec_kw={'width_ratios':[0.08,1,1,1,0.08]})
        fig.set_figheight(6)
        fig.set_figwidth(6*2.5)
        # fig.suptitle(f'Sample {sample}', fontsize = 16)
        vmin = 0
        vmax = np.mean(y)
        s1 = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap,
                        ax = ax1, cbar = False)
        s1.set_title('Experiment', fontsize = 16)
        s2 = sns.heatmap(y_pca, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap,
                        ax = ax2, cbar_ax = axcb12)
        s2.set_title(f'Max Ent with PCA(k={k})\nSCC={np.round(pca_results["scc_var"], 3)}', fontsize = 16)
        s2.set_yticks([])
        axcb12.yaxis.set_ticks_position('left')

        arr = S
        vmin = np.nanpercentile(arr, 1)
        vmax = np.nanpercentile(arr, 99)
        vmax = max(vmax, vmin * -1)
        vmin = vmax * -1
        s3 = sns.heatmap(arr, linewidth = 0, vmin = vmin, vmax = vmax, cmap = cmap2,
                        ax = ax3, cbar_ax = axcb3)
        s3.set_title(r'Max ent $\hat{S}$', fontsize = 16)
        s3.set_yticks([])

        plt.tight_layout()
        plt.savefig(osp.join(sample_dir, 'PCA_vs_Exp.png'))
        plt.close()

def plot_all_contact_maps(dataset):
    '''plot every contact map in dataset in a series of 5x5 panel images.'''
    dir = f'/home/erschultz/{dataset}/samples'
    rows = 3
    cols = 3
    fig, ax = plt.subplots(rows, cols)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig_ind = 1
    row = 0
    col = 0
    for sample in sorted(os.listdir(dir)):

        s_dir = osp.join(dir, sample)
        assert osp.exists(s_dir)
        s = int(sample[6:])
        if s < 200:
            continue
        print(sample, row, col)
        y = np.load(osp.join(s_dir, 'y.npy'))
        s = sns.heatmap(y, linewidth = 0, vmin = 0, vmax = np.mean(y), cmap = RED_CMAP,
                        ax = ax[row][col], cbar = False)
        s.set_title(sample, fontsize = 16)
        s.set_xticks([])
        s.set_yticks([])

        col += 1
        if col > cols-1:
            col = 0
            row += 1
        if row > rows-1:
            # save fit and reset
            plt.tight_layout()
            plt.savefig(osp.join(f'/home/erschultz/{dataset}/all_hic_{fig_ind}.png'))
            plt.close()

            fig, ax = plt.subplots(rows, cols)
            fig.set_figheight(12)
            fig.set_figwidth(12)
            fig_ind += 1
            row = 0
            col = 0

    # save remainder
    plt.tight_layout()
    plt.savefig(osp.join(f'/home/erschultz/{dataset}/all_hic_{fig_ind}.png'))
    plt.close()

def plot_p_s(dataset, experimental=False, ref=False, params=False, label=None):
    # plot different p(s) curves
    dir = '/home/erschultz/'
    if ref:
        data_dir = osp.join(dir, 'dataset_04_10_23/samples/sample1001') # experimental data sample
        file = osp.join(data_dir, 'y.npy')
        y_exp = np.load(file)
        meanDist_ref = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')

    data_dir = osp.join(dir, dataset)

    data = defaultdict(dict) # sample : {meanDist, diag_chis_step} : vals
    for sample in range(1, 5000):
        sample_dir = osp.join(data_dir, 'samples', f'sample{sample}')
        ifile = osp.join(sample_dir, 'y.npy')
        if osp.exists(ifile):
            y = np.load(ifile)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            data[sample]['meanDist'] = meanDist

            config_file = osp.join(sample_dir, 'config.json')
            if osp.exists(config_file):
                with open(config_file) as f:
                    config = json.load(f)
                    data[sample]['grid_size'] = config['grid_size']
                    data[sample]['phi_chromatin'] = config['phi_chromatin']
                    data[sample]['bond_length'] = config["bond_length"]
                    data[sample]['grid_size'] = config["grid_size"]
                    data[sample]['beadvol'] = config['beadvol']
                    data[sample]['k_angle'] = config['k_angle']
            if params:
                diag_chis_step = calculate_diag_chi_step(config)
                data[sample]['diag_chis_step'] = np.array(diag_chis_step)

    for norm in [True, False]:
        fig, ax = plt.subplots()
        if params:
            ax2 = ax.twinx()
        if ref:
            if norm:
                X = np.linspace(0, 1, len(meanDist_ref))
            else:
                X = np.arange(0, len(meanDist_ref), 1)
            ax.plot(X, meanDist_ref, label = 'Experiment', color = 'k')

        for i, sample in enumerate(data.keys()):
            meanDist = data[sample]['meanDist']
            if norm:
                X = np.linspace(0, 1, len(meanDist))
            else:
                X = np.arange(0, len(meanDist), 1)
            if i > 10:
                ls = 'dashed'
            else:
                ls = 'solid'

            if label is not None:
                ax.plot(X, meanDist, label = data[sample][label], ls = ls)
            else:
                ax.plot(X, meanDist, label = sample, ls = ls)

            if params:
                diag_chis_step = data[sample]['diag_chis_step']
                ax2.plot(X, diag_chis_step, ls = '--', label = 'Parameters')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Contact Probability', fontsize = 16)
        ax.set_xlabel('Polymer Distance (beads)', fontsize = 16)

        if params:
            ax.legend(loc='lower left', title = 'Sample')
            ax2.set_xscale('log')
            ax2.set_ylabel('Diagonal Parameter', fontsize = 16)
            ax2.legend(loc='upper right')
        elif label is not None:
            ax.legend(loc='upper right', title = label)
        else:
            ax.legend(loc='upper right', title = 'Sample')
        if not experimental:
            plt.title(f"b={data[sample]['bond_length']}, "
                    r"$\Delta$"
                    f"={data[sample]['grid_size']}, vb={data[sample]['beadvol']}")
        plt.tight_layout()
        plt.savefig(osp.join(data_dir, f'meanDist_norm_{norm}.png'))
        plt.close()

def plot_seq_comparison(seqs, labels):
    '''Compare sequences in seqs.

    Inputs:
        seqs: list of arrays, arrays should be mxk
        labels: list of labels
    '''
    for i, seq in enumerate(seqs):
        if seq.shape[1] > seq.shape[0]:
            seqs[i] = seq.T
    rows = 3; cols = 3
    row = 0; col = 0
    fig, ax = plt.subplots(rows, cols)
    fig.set_figheight(12)
    fig.set_figwidth(16)
    for i in range(rows*cols):
        for seq, label in zip(seqs, labels):
            ax[row, col].plot(seq[:, i], label = label)
        ax[row, col].set_title(f'PC {i+1}')
        ax[row, col].legend()

        col += 1
        if col > cols-1:
            col = 0
            row += 1
    plt.show()

def plot_energy_no_ticks():
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                         [(0, 'red'),
                                         (0.5, 'white'),
                                          (1, 'darkgreen')], N=126)
    dir = '/home/erschultz/dataset_04_28_23/samples'
    for sample in [1,2,3,4,5,324,981,1753,2834,3464]:
        dir2 = osp.join(dir, f'sample{sample}')
        # /optimize_grid_b_140_phi_0.03-max_ent10')
        # max_it_dir = get_final_max_ent_folder(dir2)
        y = np.load(osp.join(dir2, 'y.npy'))
        plot_matrix(y, osp.join(dir2, 'y_noticks.png'), vmax = 'mean',
                    x_ticks = [], y_ticks = [], use_cbar = False)

        config = load_json(osp.join(dir2, 'config.json'))
        x = np.load(osp.join(dir2, 'x.npy'))
        L, D, S = calculate_all_energy(config, x, np.array(config["chis"]))
        plot_matrix(L, osp.join(dir2, 'L_noticks.png'), cmap='bluered',
                    x_ticks = [], y_ticks = [], vmin = -15, vmax = 15,
                    use_cbar = False)
        plot_matrix(D, osp.join(dir2, 'D_noticks.png'), vmin = 5, vmax = 25,
                    cmap='bluered',
                    x_ticks = [], y_ticks = [], use_cbar = False)
        plot_matrix(S, osp.join(dir2, 'S_noticks.png'), cmap=cmap,
                    x_ticks = [], y_ticks = [], vmin = 'center', percentile=15,
                    use_cbar = False)

def generalization_figure():
    label_fontsize=24
    tick_fontsize=22
    letter_fontsize=26
    datasets = ['dataset_06_29_23']*3
    cell_lines = ['IMR90', 'HMEC', 'HAP1']
    samples = [2, 103, 604]
    # datasets = ['dataset_04_05_23', 'dataset_04_05_23', 'dataset_04_05_23']
    # cell_lines = ['GM12878', 'HCT116', 'HL-60']
    # samples = [1213, 1248, 1286]
    GNN_ID = 434

    odir = '/home/erschultz/TICG-chromatin/figures'
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    composites = []
    max_ent_composites = []
    sccs = []
    max_ent_sccs = []
    max_ent_meanDists = []
    gnn_meanDists = []
    ref_meanDists = []
    max_ent_pcs = []
    gnn_pcs = []
    ref_pcs = []
    genome_ticks_list = []
    genome_labels_list = []
    chroms = []

    # collect data
    print('---'*9)
    print('Collecting Data')
    for dataset, cell_line, sample in zip(datasets, cell_lines, samples):
        print(cell_line)
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        gnn_dir = osp.join(dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
        max_ent_dir = osp.join(dir, f'optimize_grid_b_140_phi_0.03-max_ent10_stop')

        y_exp = np.load(osp.join(dir, 'y.npy'))
        y_exp /= np.mean(y_exp.diagonal())
        ref_meanDists.append(DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob'))
        ref_pcs.append(epilib.get_pcs(epilib.get_oe(y_exp), 12, align = True).T)

        result = load_import_log(dir)
        start = result['start_mb']
        end = result['end_mb']
        chrom = result['chrom']
        resolution = result['resolution']
        chroms.append(chrom)
        m = len(y_exp)
        all_labels = np.linspace(start, end, m)
        all_labels = np.round(all_labels, 0).astype(int)
        genome_ticks = [0, m//3, 2*m//3, m-1]
        genome_labels = [f'{all_labels[i]}' for i in genome_ticks]
        genome_ticks_list.append(genome_ticks)
        genome_labels_list.append(genome_labels)

        y_gnn = np.load(osp.join(gnn_dir, 'y.npy'))
        y_gnn /= np.mean(y_gnn.diagonal())
        gnn_meanDists.append(DiagonalPreprocessing.genomic_distance_statistics(y_gnn, 'prob'))
        gnn_pcs.append(epilib.get_pcs(epilib.get_oe(y_gnn), 12, align = True).T)

        final = get_final_max_ent_folder(max_ent_dir)
        y_max_ent = np.load(osp.join(final, 'y.npy'))
        max_ent_meanDists.append(DiagonalPreprocessing.genomic_distance_statistics(y_max_ent, 'prob'))
        max_ent_pcs.append(epilib.get_pcs(epilib.get_oe(y_max_ent), 12, align = True).T)

        scc_dict = load_json(osp.join(gnn_dir, 'distance_pearson.json'))
        scc = np.round(scc_dict['scc_var'], 3)
        sccs.append(scc)

        scc_dict = load_json(osp.join(final, 'distance_pearson.json'))
        scc = np.round(scc_dict['scc_var'], 3)
        max_ent_sccs.append(scc)

        m = np.shape(y_gnn)[0]
        indu = np.triu_indices(m)
        indl = np.tril_indices(m)

        # make composite contact map
        composite = np.zeros((m, m))
        composite[indu] = y_gnn[indu]
        composite[indl] = y_exp[indl]
        np.fill_diagonal(composite, 1)
        composites.append(composite)

        composite = np.zeros((m, m))
        composite[indu] = y_max_ent[indu]
        composite[indl] = y_exp[indl]
        np.fill_diagonal(composite, 1)
        max_ent_composites.append(composite)




    ### plot hic ###
    print('---'*9)
    print('Starting Hic Figure')
    fig, axes = plt.subplots(2, len(cell_lines))
    fig.set_figheight(12)
    fig.set_figwidth(18)

    arr = np.array(composites)
    vmax = np.mean(arr)
    data = zip(composites, cell_lines, sccs, genome_ticks_list, genome_labels_list, chroms)
    for i, (composite, cell_line, scc, genome_ticks, genome_labels, chrom) in enumerate(data):
        print(i)
        ax = axes[0][i]
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = False)
        s.set_title(f'{cell_line} Chr{chrom}\nSCC={scc}', fontsize = letter_fontsize)
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, 'GNN', fontsize=letter_fontsize, ha='right', va='top',
                weight='bold')
        ax.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        # s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        # s.set_yticks(genome_ticks, labels = genome_labels)
        s.set_xticks([])
        s.set_yticks([])
        s.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    arr = np.array(max_ent_composites)
    vmax = np.mean(arr)
    data = zip(max_ent_composites, max_ent_sccs, genome_ticks_list, genome_labels_list)
    for i, (composite, scc, genome_ticks, genome_labels) in enumerate(data):
        print(i)
        ax = axes[1][i]
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = False)
        s.set_title(f'SCC={scc}', fontsize = letter_fontsize)
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, 'Max Ent', fontsize=letter_fontsize, ha='right', va='top',
                weight='bold')
        ax.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        # s.set_yticks(genome_ticks, labels = genome_labels)
        s.set_yticks([])
        s.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    for n, ax in enumerate([axes[0][0], axes[1][0]]):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_hic.png'))
    plt.close()

    ### plot meanDist ###
    fig, axes = plt.subplots(1, len(cell_lines))
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    data = zip(gnn_meanDists, max_ent_meanDists, ref_meanDists, cell_lines)
    for i, (gnn_meanDist, max_ent_meanDist, ref_meanDist, cell_line) in enumerate(data):
        print(i)
        rmse = mean_squared_error(gnn_meanDist, ref_meanDist, squared = False)
        rmse = np.round(rmse, 3)

        ax = axes[i]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.plot(ref_meanDist, label = 'Experiment', color = 'k')
        ax.plot(max_ent_meanDist, label = 'Max Ent', color = 'b')
        ax.plot(gnn_meanDist, label = 'GNN', color = 'red')
        ax.set_title(f'{cell_line}\nRMSE={rmse}', fontsize = 16)

        if i == 0:
            ax.legend()
            ax.set_ylabel('Contact Probability', fontsize=16)

    fig.supxlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_p_s.png'))
    plt.close()

    ### plot pc1 ###
    fig, axes = plt.subplots(1, len(cell_lines))
    fig.set_figheight(4)
    fig.set_figwidth(6*2.5)
    # fig.suptitle(f'Extrapolation Results', fontsize = 16)
    for i, (pc_gnn, pc_max_ent, pc_ref, cell_line) in enumerate(zip(gnn_pcs, max_ent_pcs, ref_pcs, cell_lines)):
        ax = axes[i]
        # make sure they are aligned by ensuring corr is positive
        pc_max_ent[0] *= np.sign(pearson_round(pc_ref[0], pc_max_ent[0]))
        pc_gnn[0] *= np.sign(pearson_round(pc_ref[0], pc_gnn[0]))

        ax.plot(pc_ref[0], label = 'Experiment', color = 'k')
        ax.plot(pc_gnn[0], label = 'GNN', color = 'r')
        ax.plot(pc_max_ent[0], label = 'Max Ent', color = 'b')
        ax.set_title(f'{cell_line}\nCorr(Exp, GNN)={pearson_round(pc_gnn[0], pc_ref[0])}', fontsize = 16)

        if i > 0:
            ax.set_yticks([])
        else:
            ax.legend()
            ax.set_ylabel('PC 1', fontsize=16)

    fig.supxlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_pc1.png'))
    plt.close()


    ### combined figure ###
    print('---'*9)
    print('Starting Figure')
    fig, axes = plt.subplots(3, len(cell_lines),
                            gridspec_kw={'height_ratios':[1, 0.6, 0.5]})
    fig.set_figheight(14.5)
    fig.set_figwidth(18)

    arr = np.array(composites)
    vmax = np.mean(arr)
    data = zip(composites, cell_lines, sccs, genome_ticks_list, genome_labels_list, chroms)
    for i, (composite, cell_line, scc, genome_ticks, genome_labels, chrom) in enumerate(data):
        print(i)
        ax = axes[0][i]
        # if i == len(cell_lines) - 1:
        #     s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
        #                     ax = ax, cbar_ax = ax[i+1])
        # else:
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = False)
        s.set_title(f'{cell_line} Chr{chrom}', fontsize = letter_fontsize)
        print(f'{cell_line}\nSCC={scc}')
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, 'GNN', fontsize=letter_fontsize, ha='right', va='top',
                weight='bold')
        ax.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        # s.set_yticks(genome_ticks, labels = genome_labels)
        s.set_yticks([])
        s.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    data = zip(gnn_pcs, max_ent_pcs, ref_pcs, cell_lines, genome_ticks_list, genome_labels_list)
    for i, (pc_gnn, pc_max_ent, pc_ref, cell_line, genome_ticks, genome_labels) in enumerate(data):
        ax = axes[1][i]
        ax.plot(pc_ref[0], label = 'Experiment', color = 'k')
        ax.plot(pc_max_ent[0], label = f'Max Ent (r={pearson_round(pc_max_ent[0], pc_ref[0])})', color = 'b')
        ax.plot(pc_gnn[0], label = f'GNN (r={pearson_round(pc_gnn[0], pc_ref[0])})', color = 'r')
        print(f'Max Ent (r={pearson_round(pc_max_ent[0], pc_ref[0])})', '\n',
                f'GNN (r={pearson_round(pc_gnn[0], pc_ref[0])})')
        ax.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        ax.set_yticks([])
        # ax.set_title(f'{cell_line}\nCorr(Exp, GNN)=', fontsize = 16)
        # ax.set_ylim(-0.1, 0.1)
        # ax.legend(fontsize=legend_fontsize)
        if i == 0:
            ax.set_ylabel('PC 1', fontsize=label_fontsize)

    axes[1][1].set_xlabel('Genomic Position (Mb)', fontsize = label_fontsize)


    data = zip(gnn_meanDists, max_ent_meanDists, ref_meanDists, cell_lines)
    for i, (gnn_meanDist, max_ent_meanDist, ref_meanDist, cell_line) in enumerate(data):
        print(i)
        rmse = mean_squared_error(gnn_meanDist, ref_meanDist, squared = False)
        rmse = np.round(rmse, 3)

        log_labels = np.linspace(0, resolution*(len(ref_meanDist)-1), len(ref_meanDist))
        # assumes resolution is the same for each sample

        ax = axes[2][i]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.plot(log_labels, ref_meanDist, label = 'Experiment', color = 'k')
        ax.plot(log_labels, max_ent_meanDist, label = 'Maximum Entropy', color = 'b')
        ax.plot(log_labels, gnn_meanDist, label = 'GNN', color = 'red')
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_ylim(10**-4, 1)

        # ax.set_title(f'{cell_line}\nRMSE(Exp, GNN)={rmse}', fontsize = 16)

        if i == 0:
            # ax.legend(fontsize=legend_fontsize)
            ax.set_ylabel('Contact Probability', fontsize=label_fontsize)
        else:
            ax.minorticks_off()
            ax.set_yticks([])

    axes[2][1].set_xlabel('Genomic Separation (bp)', fontsize = label_fontsize)

    for n, ax in enumerate([axes[0][0], axes[1][0], axes[2][0]]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # legend
    l4 = axes[2][1].legend(bbox_to_anchor=(0.5, -0.45), loc="upper center",
                fontsize = label_fontsize,
                borderaxespad=0, ncol=3)


    plt.savefig(osp.join(odir, 'extrapolation_figure.png'))
    plt.close()



    ### combined figure2 ###
    label_fontsize=10
    tick_fontsize=8
    letter_fontsize=10
    print('---'*9)
    print('Starting Figure')
    fig, axes = plt.subplots(3, len(cell_lines),
                            gridspec_kw={'height_ratios':[1, 0.6, 0.5]})
    fig.set_figheight(4.5)
    fig.set_figwidth(6)

    arr = np.array(composites)
    vmax = np.mean(arr)
    data = zip(composites, cell_lines, sccs, genome_ticks_list, genome_labels_list, chroms)
    for i, (composite, cell_line, scc, genome_ticks, genome_labels, chrom) in enumerate(data):
        print(i)
        ax = axes[0][i]
        # if i == len(cell_lines) - 1:
        #     s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
        #                     ax = ax, cbar_ax = ax[i+1])
        # else:
        s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = False)
        s.set_title(f'{cell_line} Chr{chrom}', fontsize = letter_fontsize)
        print(f'{cell_line}\nSCC={scc}')
        ax.axline((0,0), slope=1, color = 'k', lw=1)
        ax.text(0.99*m, 0.01*m, 'GNN', fontsize=letter_fontsize, ha='right', va='top',
                weight='bold')
        ax.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        s.set_yticks(genome_ticks, labels = genome_labels)
        s.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    data = zip(gnn_pcs, max_ent_pcs, ref_pcs, cell_lines, genome_ticks_list, genome_labels_list)
    for i, (pc_gnn, pc_max_ent, pc_ref, cell_line, genome_ticks, genome_labels) in enumerate(data):
        ax = axes[1][i]
        ax.plot(pc_ref[0], label = 'Experiment', color = 'k')
        ax.plot(pc_max_ent[0], label = f'Max Ent (r={pearson_round(pc_max_ent[0], pc_ref[0])})', color = 'b')
        ax.plot(pc_gnn[0], label = f'GNN (r={pearson_round(pc_gnn[0], pc_ref[0])})', color = 'r')
        print(f'Max Ent (r={pearson_round(pc_max_ent[0], pc_ref[0])})', '\n',
                f'GNN (r={pearson_round(pc_gnn[0], pc_ref[0])})')
        ax.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)
        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # ax.set_title(f'{cell_line}\nCorr(Exp, GNN)=', fontsize = 16)
        # ax.set_ylim(-0.1, 0.1)
        # ax.legend(fontsize=legend_fontsize)
        if i == 0:
            ax.set_ylabel('PC 1', fontsize=label_fontsize)

    axes[1][1].set_xlabel('Genomic Position (Mb)', fontsize = label_fontsize)

    data = zip(gnn_meanDists, max_ent_meanDists, ref_meanDists, cell_lines)
    for i, (gnn_meanDist, max_ent_meanDist, ref_meanDist, cell_line) in enumerate(data):
        print(i)
        rmse = mean_squared_error(gnn_meanDist, ref_meanDist, squared = False)
        rmse = np.round(rmse, 3)

        log_labels = np.linspace(0, resolution*(len(ref_meanDist)-1), len(ref_meanDist))
        # assumes resolution is the same for each sample

        ax = axes[2][i]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.plot(log_labels, ref_meanDist, label = 'Experiment', color = 'k')
        ax.plot(log_labels, max_ent_meanDist, label = 'Max Ent', color = 'b')
        ax.plot(log_labels, gnn_meanDist, label = 'GNN', color = 'red')
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # ax.set_title(f'{cell_line}\nRMSE(Exp, GNN)={rmse}', fontsize = 16)

        if i == 0:
            # ax.legend(fontsize=legend_fontsize)
            ax.set_ylabel('Contact Probability', fontsize=label_fontsize)

    axes[2][1].set_xlabel('Genomic Separation (bp)', fontsize = label_fontsize)

    for n, ax in enumerate([axes[0][0], axes[1][0], axes[2][0]]):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_figure2.png'))
    plt.close()


def figure2(test=False):
    label_fontsize=24
    tick_fontsize=22
    letter_fontsize=26
    dataset = 'dataset_02_04_23'; sample = 208; GNN_ID = 434
    # dataset = 'dataset_04_05_23'; sample = 1001; GN_ID = 407
    # dataset = 'dataset_04_05_23'; sample = 1001; GNN_ID = 423
    samples_list = [208, 209, 210, 211, 212, 213, 214, 215, 223, 224]

    k=10
    def get_dirs(sample_dir):
        grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')
        max_ent_dir = f'{grid_dir}-max_ent{k}'
        gnn_dir = f'{grid_dir}-GNN{GNN_ID}'

        return max_ent_dir, gnn_dir


    def get_y(sample_dir):
        max_ent_dir, gnn_dir = get_dirs(sample_dir)
        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))

        final = get_final_max_ent_folder(max_ent_dir)
        y_pca = np.load(osp.join(final, 'y.npy')).astype(np.float64)
        y_pca /= np.mean(np.diagonal(y_pca))

        y_gnn = np.load(osp.join(gnn_dir, 'y.npy')).astype(np.float64)
        y_gnn /= np.mean(np.diagonal(y_gnn))

        return y, y_pca, y_gnn


    sample_dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    max_ent_dir, gnn_dir = get_dirs(sample_dir)
    y, y_pca, y_gnn = get_y(sample_dir)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)
    m = len(y)

    result = load_import_log(sample_dir)
    start = result['start_mb']
    end = result['end_mb']
    chrom = result['chrom']
    resolution = result['resolution']
    resolution_mb = result['resolution_mb']
    all_labels = np.linspace(start, end, len(y))
    all_labels = np.round(all_labels, 0).astype(int)
    # genome_ticks = [0, len(y)//3, 2*len(y)//3, len(y)-1]
    genome_ticks = [0, len(y)-1]
    genome_labels = [f'{all_labels[i]}' for i in genome_ticks]

    final = get_final_max_ent_folder(max_ent_dir)
    with open(osp.join(final, 'distance_pearson.json'), 'r') as f:
        # TODO this is after final iteration, not convergence
        pca_results = json.load(f)
        pca_scc = np.round(pca_results["scc_var"], 3)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_pca)
    y_pca_diag = DiagonalPreprocessing.process(y_pca, meanDist)
    L = load_L(max_ent_dir)
    with open(osp.join(final, 'config.json'), 'r') as f:
        config = json.load(f)
    diag_chis_continuous = calculate_diag_chi_step(config)
    D = calculate_D(diag_chis_continuous)
    S = calculate_S(L, D)


    with open(osp.join(gnn_dir, 'distance_pearson.json'), 'r') as f:
        gnn_results = json.load(f)
        gnn_scc = np.round(gnn_results["scc_var"], 3)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_gnn)
    y_gnn_diag = DiagonalPreprocessing.process(y_gnn, meanDist)

    S_gnn = np.load(osp.join(gnn_dir, 'S.npy'))

    def get_pcs(input, smooth=False, h=1, verbose=False):
        if input is None:
            return None

        if smooth:
            input = scipy.ndimage.gaussian_filter(input, (h, h))

        input = epilib.get_oe(input)
        # input = np.corrcoef(input)

        if verbose:
            print(input)
        seqs = epilib.get_pcs(input, 12,
                            normalize = False, align = True)
        return seqs.T # k x m

    smooth = True; h = 1
    pcs = get_pcs(y, smooth, h)
    pcs_pca = get_pcs(y_pca, smooth, h)
    pcs_gnn = get_pcs(y_gnn, smooth, h)
    # pcs = epilib.get_pcs(epilib.get_oe(y), 12).T
    # pcs_pca = epilib.get_pcs(epilib.get_oe(y_pca), 12).T
    # pcs_gnn = epilib.get_pcs(epilib.get_oe(y_gnn), 12).T


    # make sure they are aligned by ensuring corr is positive
    pcs_pca[0] *= np.sign(pearson_round(pcs[0], pcs_pca[0]))
    pcs_gnn[0] *= np.sign(pearson_round(pcs[0], pcs_gnn[0]))

    # boxplot data
    if not test:
        args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                        samples = samples_list)
        args.experimental = True
        args.convergence_definition = None
        args.gnn_id = GNN_ID
        data, _ = load_data(args)
        max_ent = osp.split(max_ent_dir)[1]

        gnn = f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}'
        gnn_times = data[0][gnn]['total_time']
        gnn_sccs = data[0][gnn]['scc_var']
        gnn_pearsons = data[0][gnn]['pearson_pc_1']

        max_ent_times_strict = data[k][max_ent]['converged_time']
        max_ent_times_strict = [i for i in max_ent_times_strict if i is not None]
        max_ent_sccs_strict = data[k][max_ent]['scc_var']
        max_ent_sccs_strict = [i for i in max_ent_sccs_strict if not np.isnan(i)]
        max_ent_pearsons_strict = data[k][max_ent]['pearson_pc_1']

        max_ent += '_stop'
        max_ent_times = data[k][max_ent]['converged_time']
        max_ent_times = [i for i in max_ent_times if i is not None]
        max_ent_sccs = data[k][max_ent]['scc_var']
        max_ent_sccs = [i for i in max_ent_sccs if not np.isnan(i)]
        max_ent_pearsons = data[k][max_ent]['pearson_pc_1']
    else:
        max_ent_times = np.random.normal(size=100)
        max_ent_sccs = np.random.normal(loc=0.7, scale = 0.1, size=100)
        max_ent_times_strict = np.random.normal(size=100)
        max_ent_sccs_strict = np.random.normal(loc=0.75, scale = 0.05, size=100)
        max_ent_pearsons = np.random.normal(size=100)
        max_ent_pearsons_strict = np.random.normal(size=100)
        gnn_times = np.random.normal(size=100)
        gnn_sccs = np.random.normal(loc=0.6, scale = 0.12, size=100)
        gnn_pearsons = np.random.normal(size=100)


    # xyz
    file = osp.join(gnn_dir, 'production_out/output.xyz')
    xyz = xyz_load(file, multiple_timesteps=True)[::, :m, :]

    # if not test:
    #     y_chr = np.load(osp.join('/home/erschultz', dataset, f'chroms_50k/sample{chrom}/y.npy'))
    #     y_chr_diag = epilib.get_oe(y_chr)
    #     seq = epilib.get_pcs(y_chr_diag, 1, normalize = True)[:, 0]
    #     x = np.zeros((len(seq), 2))
    #     # kmeans = KMeans(n_clusters = 2)
    #     # kmeans.fit(y_chr_diag)
    #
    #     # x[np.arange(len(y_chr)), kmeans.labels_] = 1
    #
    #     # ensure that positive PC is compartment A
    #     chip_seq = np.load(f'/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/home/erschultz/chip_seq_data/GM12878/hg19/signal_p_value/ENCFF850KGH/{chrom}.npy')
    #     chip_seq = chip_seq[:, 1]
    #     corr = pearson_round(chip_seq, seq, stat = 'spearman')
    #     seq *= np.sign(corr)
    #     seq = np.sign(seq)
    #
    #     print(start / resolution_mb, end / resolution_mb)
    #     left = int(start / resolution_mb)
    #     right = int(end / resolution_mb)
    #     x = x[left:right]

    x = np.zeros((m, 2))
    y_diag = epilib.get_oe(y)
    print(y.shape, y_diag.shape)
    seq = epilib.get_pcs(y_diag, 1, normalize = True)[:, 0]
    seq_a = np.zeros_like(seq)
    seq_b = np.zeros_like(seq)
    seq_a[seq > 0] = 1
    seq_b[seq < 0] = 1

    # ensure that positive PC is compartment A based on count of A-A contacts
    # a compartment should have fewer self-self contacts as it is less dense
    count_a = seq_a @ y @ seq_a
    count_b = seq_b @ y @ seq_b
    print('self-self counts: ', count_a, count_b)
    if count_a < count_b:
        x[:, 0] = seq_a
        x[:, 1] = seq_b
    else:
        x[:, 0] = seq_b
        x[:, 1] = seq_a

    xyz_write(xyz, osp.join(gnn_dir, 'xyz.xyz'), 'w', x = x)


    ### combined figure ###
    print('---'*9)
    print('Starting Figure')
    fig = plt.figure(figsize=(18, 10.5))
    ax1 = plt.subplot(2, 24, (1, 6))
    ax2 = plt.subplot(2, 24, (9, 14))
    # ax_cb = plt.subplot(2, 48*2, 29*2-1)
    ax4 = plt.subplot(2, 24, (17, 24)) # pc
    # ax4_2 = plt.subplot(4, 24, (41, 48)) # diagonal
    ax5 = plt.subplot(2, 24, (25, 29)) # meandist
    ax6 = plt.subplot(2, 48, (63, 67))
    ax7 = plt.subplot(2, 48, (72, 76))
    ax8 = plt.subplot(2, 48, (81, 85))
    axes = [ax1, ax4, ax5, ax6, ax7, ax8]


    vmin = 0
    vmax = np.mean(y)
    # combined hic maps
    npixels = np.shape(y)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)
    data = zip([y_gnn, y_pca], ['GNN', 'Max Ent'], [ax1, ax2], [gnn_scc, pca_scc])
    for i, (y_sim, label, ax, scc) in enumerate(data):
        # make composite contact map
        composite = np.zeros((npixels, npixels))
        composite[indu] = y_sim[indu]
        composite[indl] = y[indl]

        s = sns.heatmap(composite, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = None)
        if i == 0:
            s.set_yticks(genome_ticks, labels = genome_labels,
                    fontsize = tick_fontsize)
        else:
            s.set_yticks([])
        title = (f'{label} ' + r'$\hat{H}$')
        print(title, f': SCC={scc}')
        # s.set_title(title, fontsize = label_fontsize)
        s.axline((0,0), slope=1, color = 'k', lw=1)
        s.text(0.99*m, 0.01*m, label, fontsize=letter_fontsize, ha='right', va='top', weight='bold')
        s.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
                    fontsize = tick_fontsize)


    # separate hic maps
    # s1 = sns.heatmap(y, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
    #                 ax = ax1, cbar = False)
    # s1.set_yticks(genome_ticks, labels = genome_labels)
    #
    # s1.set_title(r'Experimental Contact Map, $H$', fontsize = 16)
    # s2 = sns.heatmap(y_pca, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
    #                 ax = ax2, cbar = False)
    # title = (r'Max Ent $\hat{H}$'
    #         f' (k={k})'
    #         f'\nSCC={np.round(pca_results["scc_var"], 3)}')
    # s2.set_title(title, fontsize = 16)
    # s2.set_yticks([])
    # s3 = sns.heatmap(y_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
    #                 ax = ax3, cbar = False)
    # title = (f'GNN-{GNN_ID} '
    #         r'$\hat{H}$'
    #         f'\nSCC={np.round(gnn_results["scc_var"], 3)}')
    # s3.set_title(title, fontsize = 16)
    # s3.set_yticks([])
    #
    # for s in [s1, s2, s3]:
    #     s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)

    ax4.plot(pcs[0], label = 'Experiment', color = 'k')
    ax4.plot(pcs_pca[0], label = f'Max Ent', color = 'b')
    ax4.plot(pcs_gnn[0], label = f'GNN', color = 'r')
    print(f'Max Ent (r={pearson_round(pcs[0], pcs_pca[0])})', '\n',
            f'GNN (r={pearson_round(pcs[0], pcs_gnn[0])})')
    ax4.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
                    fontsize = tick_fontsize)
    ax4.set_yticks([])
    ax4.set_ylabel('PC 1', fontsize=label_fontsize)
    # ax4.set_title(f'PC 1\nCorr(Exp, GNN)={pearson_round(pcs[0], pcs_gnn[0])}')
    ax4.set_ylim(None, .14)
    ax4.legend(bbox_to_anchor=(0.5, .98), loc="upper center",
                fontsize = 14, borderaxespad=0, ncol=3)

    # resized = rotate_bound(y,-45)
    # height=50
    # center = resized.shape[0] // 2
    # resized = resized[center-height:center, :]
    # sns.heatmap(resized, ax = ax4_2, linewidth = 0, vmin = vmin, vmax = vmax,
    #             cmap = RED_CMAP, cbar = False)
    # ax4_2.set_xticks([])
    # ax4_2.set_yticks([])



    # compare P(s)
    log_labels = np.linspace(0, resolution*(len(meanDist)-1), len(meanDist))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    meanDist_pca = DiagonalPreprocessing.genomic_distance_statistics(y_pca, 'prob')
    meanDist_gnn = DiagonalPreprocessing.genomic_distance_statistics(y_gnn, 'prob')
    data = zip([meanDist, meanDist_pca, meanDist_gnn],
                ['Experiment', 'Max Ent', f'GNN'],
                ['k', 'b', 'r'])
    for arr, fig_label, c in data:
        print(fig_label, mean_squared_error(meanDist, arr))
        ax5.plot(log_labels, arr, label = fig_label, color = c)

    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.set_ylabel('Contact Probability', fontsize = label_fontsize)
    ax5.set_xlabel('Genomic Separation (bp)', fontsize = label_fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # ax5.legend(loc='upper right', fontsize=legend_fontsize)

    # time and scc
    labels = ['Max Ent', 'Max Ent (long)', 'GNN']
    ticks = range(1, 4)
    data = [max_ent_sccs, max_ent_sccs_strict, gnn_sccs]
    print(max_ent_sccs, np.mean(max_ent_sccs))
    print(gnn_sccs, np.mean(gnn_sccs))
    b1 = ax6.boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    ax6.set_ylim()
    ax6.set_ylabel(r'SCC(H$^{\rm sim}$, H$^{\rm exp}$)', fontsize=label_fontsize)



    data = [max_ent_pearsons_strict, max_ent_pearsons, gnn_pearsons]
    print(max_ent_pearsons_strict)
    b3 = ax7.boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    # axes[1].set_yscale('log')
    ax7.set_ylabel(r'Pearson(PC1$^{\rm sim}$, PC1$^{\rm exp}$)', fontsize=label_fontsize)

    data = [max_ent_times, max_ent_times_strict, gnn_times]
    b2 = ax8.boxplot(data,  vert = True,
                        patch_artist = True, labels = labels)
    for box in b2['boxes']:
        box.set(edgecolor='black', linewidth=4)
    # ax8.set_yticks([10, 50, 100])
    # ax8.set_yscale('log')
    ax8.set_ylim(0, 90)
    ax8.set_ylabel('Time (mins)', fontsize=label_fontsize)

    # fill with colors
    colors = ['b', 'b', 'r']
    for bplot in [b1, b2, b3]:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # rotate xticks
    # Create offset transform by 5 points in x direction
    dx = 15/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for ax in [ax6, ax7, ax8]:
        ax.set_xticks(ticks, labels, rotation=40, ha='right')
        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # axes.append(ax2)
    for n, ax in enumerate(axes):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    axes[-1].text(1.35, 1.05, string.ascii_uppercase[n+1], transform=ax.transAxes,
            size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, top = 0.95, left = 0.1, right = 0.95,
                    hspace = 0.25)
    # if test:
    #     plt.show()
    # else:
    plt.savefig('/home/erschultz/TICG-chromatin/figures/figure2.png')
    plt.close()


def interpretation_figure():
    # dataset = 'dataset_04_05_23'; sample = 1001
    dataset = 'dataset_02_04_23'; sample = 203
    sample_dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    GNN_ID = 403
    samples_list = [1001]

    y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)
    m = len(y)

    import_file = osp.join(sample_dir, 'import.log')
    with open(import_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split('=')
            if line[0] == 'start':
                start = int(line[1]) / 1000000 # Mb
            if line[0] == 'end':
                end = int(line[1]) / 1000000
    all_labels = np.linspace(start, end, len(y))
    all_labels = np.round(all_labels, 0).astype(int)
    genome_ticks = [0, len(y)//3, 2*len(y)//3, len(y)-1]
    genome_labels = [f'{all_labels[i]} Mb' for i in genome_ticks]

    grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')
    k=10
    max_ent_dir = f'{grid_dir}-max_ent{k}'
    gnn_dir = f'{grid_dir}-GNN{GNN_ID}'


    final = get_final_max_ent_folder(max_ent_dir)
    x = np.load(osp.join(final, 'x.npy'))
    L_max_ent = load_L(max_ent_dir)
    meanDist_L_max_ent = DiagonalPreprocessing.genomic_distance_statistics(L_max_ent, 'freq')
    chi_max_ent = load_max_ent_chi(k, final)
    with open(osp.join(final, 'config.json'), 'r') as f:
        config = json.load(f)
    diag_chis_continuous = calculate_diag_chi_step(config)
    D_max_ent = calculate_D(diag_chis_continuous)
    S_max_ent = calculate_S(L_max_ent, D_max_ent)


    with open(osp.join(gnn_dir, 'distance_pearson.json'), 'r') as f:
        gnn_results = json.load(f)
    y_gnn = np.load(osp.join(gnn_dir, 'y.npy')).astype(np.float64)
    y_gnn /= np.mean(np.diagonal(y_gnn))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_gnn)
    y_gnn_diag = DiagonalPreprocessing.process(y_gnn, meanDist)

    S_gnn = np.load(osp.join(gnn_dir, 'S.npy'))
    meanDist_S_gnn = DiagonalPreprocessing.genomic_distance_statistics(S_gnn, 'freq')


    # get new L
    L_max_ent = L_max_ent - calculate_D(meanDist_L_max_ent)
    L_max_ent -= np.mean(L_max_ent)
    L_gnn = S_gnn - calculate_D(meanDist_S_gnn)
    L_gnn -= np.mean(L_gnn)

    # get new x
    w, V = np.linalg.eig(L_max_ent)
    x_eig = V[:,:k]
    assert np.sum((np.isreal(x_eig)))
    x = np.real(x_eig)
    chis_eig = np.zeros_like(chi_max_ent)
    for i, val in enumerate(w[:k]):
        assert np.isreal(val)
        chis_eig[i,i] = np.real(val)

    # get new chi
    # chi_max_ent =  predict_chi_in_psi_basis(x, L_max_ent, verbose = True)
    chi_max_ent = chis_eig
    chi_gnn = predict_chi_in_psi_basis(x, L_gnn, verbose = True)

    ### combined figure ###
    plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(3, 24, (1, 6)) # S
    ax2 = plt.subplot(3, 24, (7, 12))
    ax3 = plt.subplot(3, 24, (13, 18))
    ax4 = plt.subplot(3, 24, 19)
    ax5 = plt.subplot(3, 24, (25, 30)) # chi
    ax6 = plt.subplot(3, 24, (31, 36))
    ax7 = plt.subplot(3, 24, (37, 42))
    ax8 = plt.subplot(3, 24, 43)
    ax9 = plt.subplot(3, 24, (49, 54)) # L
    ax10 = plt.subplot(3, 24, (55, 60))
    ax11 = plt.subplot(3, 24, (61, 66))
    ax12 = plt.subplot(3, 24, 67)

    arr = np.array([S_max_ent, S_gnn])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(S_max_ent, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax1, cbar = False)
    s1.set_title(f'Max Ent (k={k}) '+ r'$\hat{S}$', fontsize = 16)
    s1.set_yticks(genome_ticks, labels = genome_labels, rotation = 0)
    s2 = sns.heatmap(S_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax2, cbar = False)
    s2.set_title(f'GNN-{GNN_ID} '+r'$\hat{S}$', fontsize = 16)
    s2.set_yticks([])
    s3 = sns.heatmap(S_max_ent - S_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax3, cbar_ax = ax4)
    title = ('Difference\n'
            r'(Max Ent $\hat{S}$ - GNN $\hat{S}$)')
    s3.set_title(title, fontsize = 16)
    s3.set_yticks([])

    for s in [s1, s2, s3]:
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)

    # chis
    labels = [i for i in LETTERS[:k]]
    ticks = np.arange(k) + 0.5
    arr = np.array([chi_max_ent])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(np.triu(chi_max_ent), linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax5, cbar = False)
    s1.set_yticks(ticks, labels = labels, rotation = 0)
    s1.set_title(f'Max Ent (k={k}) '+ r'$\hat{\chi}$', fontsize = 16)
    s2 = sns.heatmap(np.triu(chi_gnn), linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax6, cbar = False)
    s2.set_yticks([])
    s2.set_title(f'GNN-{GNN_ID} '+ r'$\hat{\chi}$', fontsize = 16)
    s3 = sns.heatmap(np.triu(chi_max_ent - chi_gnn), linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax7, cbar_ax = ax8)
    s3.set_yticks([])
    title = ('Difference\n'
            r'(Max Ent $\hat{\chi}$ - GNN $\hat{\chi}$)')
    s3.set_title(title, fontsize = 16)

    for s in [s1, s2, s3]:
        s.set_xticks(ticks, labels = labels, rotation = 0)

    # L
    arr = np.array([L_max_ent, L_gnn])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(L_max_ent, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax9, cbar = False)
    s1.set_title(f'Max Ent (k={k}) '+ r'$\hat{L}$', fontsize = 16)
    s1.set_yticks(genome_ticks, labels = genome_labels, rotation = 0)
    s2 = sns.heatmap(L_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax10, cbar = False)
    s2.set_title(f'GNN-{GNN_ID} '+r'$\hat{L}$', fontsize = 16)
    s2.set_yticks([])
    s3 = sns.heatmap(L_max_ent - L_gnn, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax11, cbar_ax = ax12)
    title = ('Difference\n'
            r'(Max Ent $\hat{L}$ - GNN $\hat{L}$)')
    s3.set_title(title, fontsize = 16)
    s3.set_yticks([])

    for s in [s1, s2, s3]:
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)



    for n, ax in enumerate([ax1, ax5, ax9]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=20, weight='bold')

    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/interpretation.png')
    plt.close()

def interpretation_figure_test():
    # dataset = 'dataset_04_05_23'; sample = 1001
    dataset = 'dataset_02_04_23'; sample = 202
    sample_dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
    samples_list = [1001]

    y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)
    m = len(y)

    import_file = osp.join(sample_dir, 'import.log')
    with open(import_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split('=')
            if line[0] == 'start':
                start = int(line[1]) / 1000000 # Mb
            if line[0] == 'end':
                end = int(line[1]) / 1000000
    all_labels = np.linspace(start, end, len(y))
    all_labels = np.round(all_labels, 0).astype(int)
    genome_ticks = [0, len(y)//3, 2*len(y)//3, len(y)-1]
    genome_labels = [f'{all_labels[i]} Mb' for i in genome_ticks]

    grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')
    k=10
    max_ent_dir = f'{grid_dir}-max_ent{k}'


    final = get_final_max_ent_folder(max_ent_dir)
    x = np.load(osp.join(final, 'x.npy'))
    L_max_ent = load_L(max_ent_dir)
    chi_max_ent = load_max_ent_chi(k, final)
    with open(osp.join(final, 'config.json'), 'r') as f:
        config = json.load(f)
    diag_chis_continuous = calculate_diag_chi_step(config)
    D_max_ent = calculate_D(diag_chis_continuous)
    S_max_ent = calculate_S(L_max_ent, D_max_ent)

    meanDist_L = DiagonalPreprocessing.genomic_distance_statistics(L_max_ent, 'freq')
    meanDist_S = DiagonalPreprocessing.genomic_distance_statistics(S_max_ent, 'freq')


    # this is relatively close
    L_max_ent -= np.mean(L_max_ent)
    L_max_ent_hat = S_max_ent - np.mean(S_max_ent)


    # this give the same result exactly
    # L_max_ent = L_max_ent - calculate_D(meanDist_L)
    # L_max_ent -= np.mean(L_max_ent)
    # L_max_ent_hat = S_max_ent - calculate_D(meanDist_S)
    # L_max_ent_hat -= np.mean(L_max_ent_hat)

    # recalculate chi
    chi_max_ent =  predict_chi_in_psi_basis(x, L_max_ent, verbose = True)
    chi_max_ent_hat = predict_chi_in_psi_basis(x, L_max_ent_hat, verbose = False)


    ### combined figure ###
    plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(2, 24, (1, 6)) # L
    ax2 = plt.subplot(2, 24, (7, 12))
    ax3 = plt.subplot(2, 24, (13, 18))
    ax4 = plt.subplot(2, 24, 19)
    ax5 = plt.subplot(2, 24, (25, 30)) # chi
    ax6 = plt.subplot(2, 24, (31, 36))
    ax7 = plt.subplot(2, 24, (37, 42))
    ax8 = plt.subplot(2, 24, 43)

    arr = np.array([L_max_ent, L_max_ent_hat])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(L_max_ent, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax1, cbar = False)
    s1.set_title(f'Max Ent (k={k}) L', fontsize = 16)
    s1.set_yticks(genome_ticks, labels = genome_labels, rotation = 0)
    s2 = sns.heatmap(L_max_ent_hat, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax2, cbar = False)
    s2.set_title(f'Max Ent (k={k}) '+r'$\hat{L}$', fontsize = 16)
    s2.set_yticks([])
    diff = L_max_ent - L_max_ent_hat
    s3 = sns.heatmap(diff, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax3, cbar_ax = ax4)
    title = (f'Difference\n left - right (mean={np.round(np.mean(np.abs(diff)), 2)})')
    s3.set_title(title, fontsize = 16)
    s3.set_yticks([])

    for s in [s1, s2, s3]:
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0)

    # chis
    labels = [i for i in LETTERS[:k]]
    ticks = np.arange(k) + 0.5
    arr = np.array([chi_max_ent])
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    vmax = max(vmax, vmin * -1)
    vmin = vmax * -1
    s1 = sns.heatmap(np.triu(chi_max_ent), linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax5, cbar = False)
    s1.set_yticks(ticks, labels = labels, rotation = 0)
    s1.set_title(f'Max Ent (k={k}) '+ r'$\chi$', fontsize = 16)
    s2 = sns.heatmap(np.triu(chi_max_ent_hat), linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax6, cbar = False)
    s2.set_yticks([])
    s2.set_title(f'Max Ent (k={k}) '+ r'$\hat{\chi}$', fontsize = 16)
    s3 = sns.heatmap(np.triu(chi_max_ent - chi_max_ent_hat), linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_BLUE_CMAP,
                    ax = ax7, cbar_ax = ax8)
    s3.set_yticks([])


    for s in [s1, s2, s3]:
        s.set_xticks(ticks, labels = labels, rotation = 0)

    for n, ax in enumerate([ax1, ax5]):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes,
                size=20, weight='bold')

    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/interpretation_test.png')
    plt.close()



if __name__ == '__main__':
    # plot_diag_vs_diag_chi()
    # plot_xyz_gif_wrapper()
    # plot_centroid_distance(parallel = True, samples = [34, 35, 36])
    # update_result_tables('ContactGNNEnergy', 'GNN', 'energy')

    # data_dir = osp.join(dir, 'dataset_soren/samples/sample1')
    # file = osp.join(data_dir, 'y_kr.npy')
    # data_dir = osp.join(dir, 'dataset_07_20_22/samples/sample4')
    # file = osp.join(data_dir, 'y.npy')
    # plot_mean_vs_genomic_distance_comparison('/home/erschultz/dataset_test_diag1024_linear', [21, 23, 25 ,27], ref_file = file)
    # plot_combined_models('ContactGNNEnergy', [427, 429])
    # plot_GNN_vs_PCA('dataset_04_05_23', 10, 407)
    # plot_first_PC('dataset_02_04_23/samples/sample202/PCA-normalize-E/k8/replicate1', 8, 392)
    # plot_Exp_vs_PCA("dataset_02_04_23")
    # main()
    # plot_all_contact_maps('dataset_05_28_23')
    # plot_p_s('dataset_05_28_23', ref=True)
    generalization_figure()
    figure2()
    # interpretation_figure()
    # interpretation_figure_test()
    # plot_first_PC('dataset_02_04_23', 10, 419)
    # plot_seq_comparison([np.load('/home/erschultz/dataset_02_04_23/samples/sample203/optimize_grid_b_16.5_phi_0.06-max_ent/iteration15/x.npy')], ['max_ent'])
    # plot_energy_no_ticks()
