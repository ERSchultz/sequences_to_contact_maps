import csv
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from data_generation.modify_maxent import get_samples
from makeLatexTable import *


def main():
    descr_dict = {434: 'baseline',
            # 472: r'predict $S$',
            441: 'without $\mean(\diagonal(H^b, |i-j|))$ in $e_{ij}$',
            442: "without overwriting main diagonal with 1's",
            443: 'without SignNet', # (eigenvectors are still included as node features)
            444: 'without $\log(H_{ij})$ in $e_{ij}$',
            445: 'without $\mean(\diagonal(H, |i-j|))$ in $e_{ij}$',
            446: 'only long simulation',
            # 447: 'without SignNet (eigenvectors are replaced with constant node feature)',
            448: 'original message passing layer from \citep{Brody2022HowNetworks}',
            449: 'without rescaling contact map'
            }
    descr_dict = {490: 'baseline 2-3-400k sweeps',
            505: 'N=5000 samples',
            507: 'baseline 2-3-4-500k sweeps',
            506: 'N=20000',
            508: 'latent dim=128',
            509: 'only 500k sweeps',
            510: 'more MP layers, lr=1e-5',
            511: 'deeper update layer',
            512: 'without SignNet', # (eigenvectors are still included as node features)
            513: 'rescale=1, only 500k sweeps, lr=1e-5',
            514: 'rescale=1',
            515: 'rescale=1, only 500k sweeps',
            516: 'more MP layers',
            517: 'lr=1e-5',
            520: '+params in hiddenSizesList, deeper update layer, +1 MP layer',
            521: 'sweepchoices = 3-4-5-600k',
            522: 'rescale=1, 4-5-600k sweeps'
            }
    descr_dict = {496: 'baseline 2-3-4k sweeps',
            518: 'baseline 2-3-4-5k sweeps',
            519: '+params in hiddenSizesList, deeper update layer, +1 MP layer',
            523: 's_1_cutoff_0.36',
            524: 's_10_cutoff_0.08',
            525: 's_100_cutoff_0.01'
            }
    descr_dict = {579: 'Baseline',
            590: 'no log-loss',
            591: 'no rescale',
            # 592: 'without SignNet and without eigenvectors',
            593: 'without $\mean(\diagonal(H, |i-j|))$ in $e_{ij}$',
            594: 'without $\mean(\diagonal(H^b, |i-j|))$ in $e_{ij}$',
            595: 'without $Corr(\\tilde{H}^\prime)_{ij}$',
            596: '$H$_{ij} instead of $Corr(\\tilde{H}^\prime)_{ij}$',
            597: 'without SignNet but with eigenvectors',
            598: 'original message passing layer from \citep{Brody2022HowNetworks}'}
    descr_dict = {631: 'Baseline',
            632: 'no log-transform in loss',
            # 633: 'no rescale',
            634: 'without SignNet and without eigenvectors',
            # 635: 'without $\mean(\diagonal(H, |i-j|))$ in $e_{ij}$',
            # 637: 'without $H_{ij}$',
            638: '$Corr(\\tilde{H}_{ij})$ instead of $H_{ij}$',
            # 636: 'without SignNet but with eigenvectors',
            # 639: 'pReLU activations',
            # 640: 'original message passing layer from \citep{Brody2022HowNetworks}',
            }
    descr_dict = {631: 'MSE($S^\dag, \hat{S}^\dag$) (baseline)',
                # 632: 'MSE($S,\hat{S}$)',
                # 641: 'MSE($S^\dag, \hat{S}^\dag$) + 0.1MSE($S,\hat{S}$)',
                # 642: 'MSE($S^\dag, \hat{S}^\dag$) + 0.05MSE($S,\hat{S}$)',
                # 643: 'MSE($S^\dag, \hat{S}^\dag$) + 0.001MSE($S,\hat{S}$)',
                # 644: 'mse_log_and_mse_kth_diagonal',
                # 645: 'mse_log_and_mse_kth_diagonal',
                # 646: '$\sum(S^\dag-\hat{S}^\dag)\odot W^\\text{SCC})^2 / m^2$',
                # 647: 'MSE($S^\dag, \hat{S}^\dag$) + $\sum(S^\dag-\hat{S}^\dag)\odot W^\\text{SCC})^2 / m^2$',
                # 648: 'MSE($S^\dag, \hat{S}^\dag$) + $0.1\sum(S^\dag-\hat{S}^\dag)\odot W^\\text{SCC})^2 / m^2$',
                # 649: 'MSE($L^\dag, \hat{L}^\dag$) + MSE($D^\dag, \hat{D}^\dag$)',
                # 650: 'MSE($S^\dag, \hat{S}^\dag$) + MSE($L^\dag, \hat{L}^\dag$) + MSE($D^\dag, \hat{D}^\dag$)',
                # 651: 'MSE($S^\dag, \hat{S}^\dag$) + MSE($(VSV^T)^\dag, (V\hat{S}V^T$)^\dag)',
                # 652: 'MSE($S^\dag, \hat{S}^\dag$) + 0.1MSE($(VSV^T)^\dag, (V\hat{S}V^T$)^\dag)',
                # 653: 'MSE($S^\dag, \hat{S}^\dag$) + 0.1MSE($VSV^T, V\hat{S}V^T$)',
                # 654: 'MSE($S^\dag, \hat{S}^\dag$) + 0.01MSE($VSV^T, V\hat{S}V^T$)',
                # 655: 'SCC($S, \hat{S}$)',
                # 656: 'MSE($S^\dag, \hat{S}^\dag$) + SCC($S, \hat{S}$)'
                # 657: 'MSE($S^\dag, \hat{S}^\dag$) (batch size=2)',
                # 658: 'MSE($S^\dag, \hat{S}^\dag$) (batch size = 4)',
                # 662: 'input L to D (meanDist)',
                # 663: 'input L to D (subtract)',
                # 664: 'input L to D (meandist_eigval)',
                # 665: "dconv in head architecture",
                # 666: "MSE($S^\dag, \hat{S}^\dag$) + scc loss no resize"
                # 667: "gradient clipping (1)",
                # 668: 'MSE($S^\dag, \hat{S}^\dag$) + 0.1SCC($S, \hat{S}$)',
                # 669: 'big dataset',
                # 672: 'SCC($\exp{-S}, \exp{-\hat{S}}$) (clip15 + norm)',
                # 673: "new dataset",
                # 674: 'batch size = 4, gradient clipping (1)',
                # 675: 'batch size = 6',
                # 676: 'new dataset, grad clip (1)',
                # 677: 'new dataset, batch size 2',
                # 678: 'new dataset (10k)',
                # 679: 'pretrain new large dataset',
                # 680: 'new dataset (10k) early stopping (40 epochs)',
                # 681: 'new dataset (10k) new random seed',
                # 682: 'correct start 2 dataset',
                # 683: 'correct start 1 dataset',
                # 684: '631 fine tune',
                # 685: '631 fine tune (exp)',
                # 686: '673 fine tune',
                # 687: '673 fine tune (exp)',
                # 688: 'variant start 1 dataset',
                # 689: 'variant start 2 dataset',
                690: 'variant start 2 dataset (10k)',
                691: '631 fine tune (exp2)',
                692: '673 fine tune (exp2)',
                693: '631 fine tune (imr90_exp)',
                694: '673 fine tune (imr90_exp)',
                }
    id_list = descr_dict.keys()
    print(id_list)


    loss_dict = defaultdict(lambda: None)
    for id in descr_dict.keys():
        gnn_dir = osp.join('/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy', str(id))
        log_file = osp.join(gnn_dir, 'out.log')
        if osp.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if line.startswith('Final val loss: '):
                        final_val_loss = np.round(float(line.split(':')[1].strip()), 3)
                        break
                else:
                    final_val_loss = None
                    print(f'Warning: loss not found for {log_file}')
            loss_dict[id] = final_val_loss
    print(loss_dict)

    dataset = 'dataset_12_06_23'
    train_samples, _ = get_samples(dataset, train=True, filter_cell_lines=['imr90'])
    train_samples = train_samples
    args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                    samples = train_samples)
    args.experimental = True
    args.bad_methods = ['b_140', 'b_261', 'spheroid_2.0', 'max_ent', 'grid200', 'phi', '_xyz']
    args.convergence_definition = 'normal'
    args.gnn_id = id_list
    data, _ = load_data(args)

    keys = ['scc_var', 'hic_spector', 'pearson_pc_1', 'rmse-ydiag']
    scc_dict = defaultdict(lambda: None)
    spector_dict = defaultdict(lambda: None)
    corr_dict = defaultdict(lambda: None)
    rmse_dict = defaultdict(lambda: None)
    dicts = [scc_dict, spector_dict, corr_dict, rmse_dict]
    labels = ['SCC', 'HiC-Spector', 'Corr PC 1', r'RMSE($\tilde{H}$)']
    for id in descr_dict.keys():
        gnn = f'optimize_grid_b_200_v_8_spheroid_1.5-GNN{id}'
        for key, data_dict in zip(keys, dicts):
            vals = data[0][gnn][key]
            data_dict[id] = np.round(np.mean(vals), 3)

    make_latex_table(id_list, descr_dict, loss_dict, dicts, labels)

def make_latex_table(id_list, descr_dict, loss_dict, dicts, labels):
    ofile = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/ablation.txt'
    with open(ofile, 'w') as o:
        # set up first rows of table
        o.write("\\renewcommand\\theadalign{bc}\n")
        o.write("\\renewcommand\\theadfont{\\bfseries}\n")
        o.write("\\begin{table}[h]\n")
        o.write("\t\\centering\n")
        num_cols = 3 + len(labels)
        num_cols_str = str(num_cols)

        o.write("\t\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        # o.write("\\hline\n")
        # o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        o.write("\t\t\\hline\n")

        row = "\t\t \\thead{ID} & \\thead{Method} & \\thead{Loss}"
        for label in labels:
            row += f" & \\thead\u007b{label}\u007d"
        row += " \\\ \n"
        o.write(row)
        o.write("\t\t\\hline\\hline\n")

        for id in id_list:
            o.write(f"\t\t {id} & {descr_dict[id]} & {loss_dict[id]}")
            for data_dict in dicts:
                o.write(f" & {data_dict[id]}")
            o.write(" \\\ \n")
        o.write("\t\t\\hline\n")
        o.write("\t\\end{tabular}\n")
        o.write('''\t\\caption{Neural network ablation results.
        See \\nameref{ablation_study} for description of each method.
        All results in the body of the paper correspond to the baseline method.
        Validation Loss (MSE) is the average mean squared error on the validation
        set of 500 simulated contact maps and their synthetic parameters.
        Experimental SCC is the average stratum-adjusted correlation (SCC)
        between the experimental contact map and a contact map simulated using
        the GNN-predicted parameters, averaged over 10 experimental contact maps.
        }\n''')
        o.write("\t\\label{table:ablation}\n")
        o.write("\\end{table}")


if __name__ == '__main__':
    main()
