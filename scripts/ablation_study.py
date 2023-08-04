import csv
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from makeLatexTable_new import *


def main():
    descr_dict = {434: 'baseline',
            440: r'predict $S$',
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


    odd_samples = [201, 202, 203, 204, 205, 206, 207, 216, 217, 218]
    scc_dict = defaultdict(lambda: None)
    args = getArgs(data_folder = f'/home/erschultz/dataset_02_04_23',
                    samples = odd_samples)
    args.experimental = True
    args.convergence_definition = 'normal'
    data, _ = load_data(args)
    for id in descr_dict.keys():
        gnn = f'optimize_grid_b_140_phi_0.03-GNN{id}'
        sccs = data[0][gnn]['scc_var']
        scc_dict[id] = np.round(np.mean(sccs), 3)

    make_latex_table(descr_dict, loss_dict, scc_dict)

def make_latex_table(descr_dict, loss_dict, scc_dict):
    ofile = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/ablation.txt'
    with open(ofile, 'w') as o:
        # set up first rows of table
        o.write("\\renewcommand\\theadalign{bc}\n")
        o.write("\\renewcommand\\theadfont{\\bfseries}\n")
        o.write("\\begin{table}[h]\n")
        o.write("\t\\centering\n")
        num_cols = 3
        num_cols_str = str(num_cols)

        o.write("\t\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        # o.write("\\hline\n")
        # o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        o.write("\t\t\\hline\n")

        row = "\t\t\\thead{Method} & \\thead{Validation Loss \\\ (MSE)} & \\thead{Test Loss \\\ (Experimental SCC)} \\\ \n"
        o.write(row)
        o.write("\t\t\\hline\\hline\n")

        id_list = [434, 440, 448, 442, 443, 449, 446, 444, 445, 441]
        for id in id_list:
            o.write(f"\t\t{descr_dict[id]} & {loss_dict[id]} & {scc_dict[id]} \\\ \n")
            if id == 427:
                o.write("\t\t\\hline\n")
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
