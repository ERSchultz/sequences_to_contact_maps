import csv
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from data_generation.modify_maxent import get_samples
from makeLatexTable_new import *


def main():
    # descr_dict = {434: 'baseline',
    #         # 472: r'predict $S$',
    #         441: 'without $\mean(\diagonal(H^b, |i-j|))$ in $e_{ij}$',
    #         442: "without overwriting main diagonal with 1's",
    #         443: 'without SignNet', # (eigenvectors are still included as node features)
    #         444: 'without $\log(H_{ij})$ in $e_{ij}$',
    #         445: 'without $\mean(\diagonal(H, |i-j|))$ in $e_{ij}$',
    #         446: 'only long simulation',
    #         # 447: 'without SignNet (eigenvectors are replaced with constant node feature)',
    #         448: 'original message passing layer from \citep{Brody2022HowNetworks}',
    #         449: 'without rescaling contact map'
    #         }
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
            592: 'without SignNet and without eigenvectors',
            593: 'without $\mean(\diagonal(H, |i-j|))$ in $e_{ij}$',
            594: 'without $\mean(\diagonal(H^b, |i-j|))$ in $e_{ij}$',
            595: 'without $H$_corr',
            596: '$H$ instead of $H$_corr',
            597: 'without SignNet but with eigenvectors'}
            598: 'original message passing layer from \citep{Brody2022HowNetworks}',
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

    dataset = 'dataset_02_04_23'
    train_samples, _ = get_samples(dataset, train=True)
    scc_dict = defaultdict(lambda: None)
    args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                    samples = train_samples[:10])
    args.experimental = True
    args.bad_methods = ['b_140', 'b_261', 'spheroid_2.0', 'max_ent']
    args.convergence_definition = 'normal'
    args.gnn_id = id_list
    data, _ = load_data(args)
    for id in descr_dict.keys():
        gnn = f'optimize_grid_b_180_phi_0.008_spheroid_1.5-GNN{id}'
        sccs = data[0][gnn]['scc_var']
        scc_dict[id] = np.round(np.mean(sccs), 3)

    make_latex_table(id_list, descr_dict, loss_dict, scc_dict)

def make_latex_table(id_list, descr_dict, loss_dict, scc_dict):
    ofile = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/ablation.txt'
    with open(ofile, 'w') as o:
        # set up first rows of table
        o.write("\\renewcommand\\theadalign{bc}\n")
        o.write("\\renewcommand\\theadfont{\\bfseries}\n")
        o.write("\\begin{table}[h]\n")
        o.write("\t\\centering\n")
        num_cols = 4
        num_cols_str = str(num_cols)

        o.write("\t\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        # o.write("\\hline\n")
        # o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        o.write("\t\t\\hline\n")

        row = "\t\t \\thead{ID} & \\thead{Method} & \\thead{Validation Loss \\\ (MSE)} & \\thead{Test Loss \\\ (Experimental SCC)} \\\ \n"
        o.write(row)
        o.write("\t\t\\hline\\hline\n")

        for id in id_list:
            o.write(f"\t\t {id} & {descr_dict[id]} & {loss_dict[id]} & {scc_dict[id]} \\\ \n")
        o.write("\t\t\\hline\n")
        o.write("\t\\end{tabular}\n")
        o.write('''\t\\caption{Neural network ablation results.
        See \\nameref{ablation_study} for description of each method.
        All results in the body of the paper correspond to the baseline method.
        Validation Loss (MSE) is the average mean squared error on the validation set of 500 simulated simulated conatact maps and their synthetic parameters.
        Experimental SCC is the average stratum adjusted correlation (SCC) between the experimental contact map and a contact map simulated using the GNN-predicted parameters, averaged over 10 experimental contact maps.
        }\n''')
        o.write("\t\\label{table:ablation}\n")
        o.write("\\end{table}")


if __name__ == '__main__':
    main()
