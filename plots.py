import csv
import json
import math
import os
import os.path as osp
import sys
from shutil import rmtree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pylib.utils import epilib
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_S)
from pylib.utils.plotting_utils import RED_CMAP, plot_matrix
from pylib.utils.utils import load_json
from scipy.ndimage import uniform_filter
from scripts.argparse_utils import (finalize_opt, get_base_parser,
                                    get_opt_header, opt2list)
from scripts.load_utils import (get_final_max_ent_folder, load_contact_map,
                                load_L)
from scripts.plotting_utils import (BLUE_RED_CMAP, RED_CMAP,
                                    plot_centroid_distance,
                                    plot_combined_models, plot_diag_chi,
                                    plot_sc_contact_maps, plot_xyz_gif,
                                    plotting_script)
from scripts.utils import DiagonalPreprocessing, pearson_round
from scripts.xyz_utils import xyz_load, xyz_write
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


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
    dir = '/home/erschultz/consistency_check/baseline_energy_on'
    file = osp.join(dir, 'production_out/output.xyz')

    m=512

    # x = np.load(osp.join(dir, 'x.npy'))[:m, :]
    x = np.arange(1, m+1) # use this to color by m
    xyz = xyz_load(file, multiple_timesteps=True)[::, :m, :]
    print(xyz.shape)
    xyz_write(xyz, osp.join(dir, 'production_out/output_x.xyz'), 'w', x = x)

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
    for sample in range(1002, 1003):
        sample_dir = osp.join(dir, f'sample{sample}')
        if not osp.exists(sample_dir):
            print(f'sample_dir does not exist: {sample_dir}')
            continue

        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))
        meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
        y_diag = DiagonalPreprocessing.process(y, meanDist)

        grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')
        max_ent_dir = f'{grid_dir}-max_ent'
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
    for sample in range(201, 210):
        sample_dir = osp.join(dir, f'sample{sample}')
        if not osp.exists(sample_dir):
            print(f'sample_dir does not exist: {sample_dir}')
            continue

        y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
        y /= np.mean(np.diagonal(y))

        grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')

        max_ent_dir = f'{grid_dir}-max_ent'
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

        rows = 2; cols = 1
        row = 0; col = 0
        fig, ax = plt.subplots(rows, cols)
        fig.set_figheight(12)
        fig.set_figwidth(16)
        for i in range(rows*cols):
            ax[row].plot(pcs[i], label = 'Experiment', color = 'k')
            ax[row].plot(pcs_pca[i], label = 'Max Ent', color = 'b')
            ax[row].plot(pcs_gnn[i], label = f'GNN-{GNN_ID}', color = 'r')
            ax[row].set_title(f'PC {i+1}\nCorr(Exp, GNN)={pearson_round(pcs[i], pcs_gnn[i])}')
            ax[row].legend()

            col += 1
            if col > cols-1:
                col = 0
                row += 1
        plt.savefig(osp.join(sample_dir, 'pc_comparison.png'))


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
        print(sample, row, col)
        s_dir = osp.join(dir, sample)
        assert osp.exists(s_dir)
        s = int(sample[6:])
        # if s < 200:
        #     continue
        y = np.load(osp.join(s_dir, 'y.npy'))
        s = sns.heatmap(y, linewidth = 0, vmin = 0, vmax = np.mean(y), cmap = RED_CMAP,
                        ax = ax[row][col], cbar = False)
        # s.set_title(sample, fontsize = 16)
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
        data_dir = osp.join(dir, 'dataset_11_14_22/samples/sample1') # experimental data sample
        file = osp.join(data_dir, 'y.npy')
        y_exp = np.load(file)
        meanDist_ref = DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob')

    data_dir = osp.join(dir, dataset)

    data = defaultdict(dict) # sample : {meanDist, diag_chis_step} : vals
    samples, _ = get_samples(dataset)
    for sample in samples:
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
                X = np.arange(0, 1, len(meanDist_ref))
            else:
                X = np.arange(0, len(meanDist_ref), 1)
            ax.plot(meanDist_ref, label = 'Experiment', color = 'k')

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
    dir = '/home/erschultz/dataset_02_04_23/samples'
    for sample in [211, 213, 218]:
        dir2 = osp.join(dir, f'sample{sample}/optimize_grid_b_140_phi_0.06-max_ent')
        max_it_dir = get_final_max_ent_folder(dir2)
        config = load_json(osp.join(max_it_dir, 'config.json'))
        x = np.load(osp.join(dir2, 'resources/x.npy'))
        L, D, S = calculate_all_energy(config, x, np.array(config["chis"]))
        plot_matrix(L, osp.join(dir2, 'L_noticks.png'), cmap='bluered',
                    x_ticks = [], y_ticks = [], vmin = -15, vmax = 15)
        plot_matrix(D, osp.join(dir2, 'D_noticks.png'), vmin = 5, vmax = 25,
                    cmap='bluered',
                    x_ticks = [], y_ticks = [])
        plot_matrix(S, osp.join(dir2, 'S_noticks.png'), cmap='bluered',
                    x_ticks = [], y_ticks = [])

def compare_different_cell_lines():
    datasets = ['Su2020', 'dataset_02_04_23', 'dataset_HCT116']
    cell_lines = ['IMR90', 'GM12878', 'HCT116']
    samples = ['1002', '202', '1010']
    GNN_ID = 403

    odir = '/home/erschultz/TICG-chromatin/figures'
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    composites = []
    sccs = []
    max_ent_meanDists = []
    gnn_meanDists = []
    ref_meanDists = []
    max_ent_pcs = []
    gnn_pcs = []
    ref_pcs = []

    for dataset, cell_line, sample in zip(datasets, cell_lines, samples):
        dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
        gnn_dir = osp.join(dir, f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}')
        max_ent_dir = osp.join(dir, f'optimize_grid_b_140_phi_0.03-max_ent')

        y_exp = np.load(osp.join(dir, 'y.npy'))
        ref_meanDists.append(DiagonalPreprocessing.genomic_distance_statistics(y_exp, 'prob'))
        ref_pcs.append(epilib.get_pcs(epilib.get_oe(y_exp), 12, align = True).T)

        y_gnn = np.load(osp.join(gnn_dir, 'y.npy'))
        gnn_meanDists.append(DiagonalPreprocessing.genomic_distance_statistics(y_gnn, 'prob'))
        gnn_pcs.append(epilib.get_pcs(epilib.get_oe(y_gnn), 12, align = True).T)

        final = get_final_max_ent_folder(max_ent_dir)
        y_max_ent = np.load(osp.join(final, 'y.npy'))
        max_ent_meanDists.append(DiagonalPreprocessing.genomic_distance_statistics(y_max_ent, 'prob'))
        max_ent_pcs.append(epilib.get_pcs(epilib.get_oe(y_max_ent), 12, align = True).T)

        scc_dict = load_json(osp.join(gnn_dir, 'distance_pearson.json'))
        print(scc_dict)
        scc = np.round(scc_dict['scc_var'], 3)
        sccs.append(scc)

        m = np.shape(y_gnn)[0]
        indu = np.triu_indices(m)
        indl = np.tril_indices(m)

        # make composite contact map
        composite = np.zeros((m, m))
        composite[indu] = y_gnn[indu]
        composite[indl] = y_exp[indl]

        composites.append(composite)


    ### plot hic ###
    fig, ax = plt.subplots(1, len(cell_lines)+1, gridspec_kw={'width_ratios':[1,1,1,0.08]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    # fig.suptitle(f'Extrapolation Results', fontsize = 16)

    arr = np.array(composites)
    vmax = np.mean(arr)
    for i, (composite, cell_line, scc) in enumerate(zip(composites, cell_lines, sccs)):
        print(i)
        if i == len(cell_lines) - 1:
            s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                            ax = ax[i], cbar_ax = ax[i+1])
        else:
            s = sns.heatmap(composite, linewidth = 0, vmin = 0, vmax = vmax, cmap = RED_CMAP,
                            ax = ax[i], cbar = False)
        s.set_title(f'{cell_line}\nSCC={scc}', fontsize = 16)
        ax[i].axline((0,0), slope=1, color = 'k', lw=1)
        ax[i].text(0.99*m, 0.01*m, 'Simulation', fontsize=16, ha='right', va='top')
        ax[i].text(0.01*m, 0.99*m, 'Experiment', fontsize=16)

        if i > 0:
            s.set_yticks([])

    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_hic.png'))
    plt.close()

    ### plot meanDist ###
    fig, axes = plt.subplots(1, len(cell_lines), gridspec_kw={'width_ratios':[1,1,1]})
    fig.set_figheight(6)
    fig.set_figwidth(6*2.5)
    for i, (gnn_meanDist, ref_meanDist, cell_line) in enumerate(zip(gnn_meanDists, ref_meanDists, cell_lines)):
        print(i)
        rmse = mean_squared_error(gnn_meanDist, ref_meanDist, squared = False)
        rmse = np.round(rmse, 3)

        ax = axes[i]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.plot(ref_meanDist, label = 'Experiment', color = 'k')
        ax.plot(gnn_meanDist, label = 'Simulation', color = 'red')
        ax.set_title(f'{cell_line}\nRMSE={rmse}', fontsize = 16)

        if i > 0:
            ax.set_yticks([])
        else:
            ax.legend()
            ax.set_ylabel('Contact Probability', fontsize=16)

    fig.supxlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_p_s.png'))
    plt.close()

    ### plot pc1 ###
    fig, axes = plt.subplots(1, len(cell_lines), gridspec_kw={'width_ratios':[1,1,1]})
    fig.set_figheight(4)
    fig.set_figwidth(6*2.5)
    # fig.suptitle(f'Extrapolation Results', fontsize = 16)
    for i, (pc_gnn, pc_max_ent, pc_ref, cell_line) in enumerate(zip(gnn_pcs, max_ent_pcs, ref_pcs, cell_lines)):
        ax = axes[i]
        ax.plot(pc_ref[0], label = 'Experiment', color = 'k')
        ax.plot(pc_gnn[0], label = 'GNN', color = 'r')
        ax.plot(pc_max_ent[0], label = 'Max Ent', color = 'b')
        ax.set_title(f'{cell_line}\nCorr(Exp, GNN)={pearson_round(pc_gnn[0], pc_ref[0])}', fontsize = 16)

        if i > 0:
            ax.set_yticks([])
        else:
            ax.legend()
            ax.set_ylabel('Value', fontsize=16)

    fig.supxlabel('Polymer Distance (beads)', fontsize = 16)
    plt.tight_layout()
    plt.savefig(osp.join(odir, 'extrapolation_pc1.png'))
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
    # plot_combined_models('ContactGNNEnergy', [400, 401])
    plot_GNN_vs_PCA('Su2020', 10, 403)
    # plot_first_PC('dataset_02_04_23/samples/sample202/PCA-normalize-E/k8/replicate1', 8, 392)
    # plot_Exp_vs_PCA("dataset_02_04_23")
    # main()
    # plot_all_contact_maps('dataset_02_16_23')
    # compare_different_cell_lines()
    # plot_first_PC('dataset_02_04_23', 10, 403)
    # plot_seq_comparison([np.load('/home/erschultz/dataset_02_04_23/samples/sample203/optimize_grid_b_16.5_phi_0.06-max_ent/iteration15/x.npy')], ['max_ent'])
