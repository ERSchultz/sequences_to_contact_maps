import csv
import os.path as osp
from collections import defaultdict

import numpy as np


def main():
    descr_dict = {427: 'Baseline',
            430: r'predict $S$ (instead of $S^\dag$)',
            431: 'without $\mean(\diagonal(H^b, |i-j|))$ in $e_{ij}$',
            432: 'without $\log(H_{ij})$ in $e_{ij}$',
            433: 'without $\mean(\diagonal(H, |i-j|))$ in $e_{ij}$',
            434: 'without $|i-j|$ in $e_{ij}$',
            435: 'without SignNet (eigenvectors are replaced with constant node feature)',
            436: 'original message passing layer from \citep{Brody2022HowNetworks} (instead of \cref{eq:mp})',
            437: "without overwriting main diagonal with 1's",
            438: 'without SignNet (eigenvectors are still included as node features)',
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

    make_latex_table(descr_dict, loss_dict)

def make_latex_table(descr_dict, loss_dict):
    ofile = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/ablation.txt'
    with open(ofile, 'w') as o:
        # set up first rows of table
        o.write("\\begin{table}[h]\n")
        o.write("\t\\centering\n")
        num_cols = 2
        num_cols_str = str(num_cols)

        o.write("\t\\begin{tabular}{|" + "c|"*num_cols + "}\n")
        # o.write("\\hline\n")
        # o.write("\\multicolumn{" + num_cols_str + "}{|c|}{" + header + "} \\\ \n")
        o.write("\t\t\\hline\n")

        row = "\t\tMethod & Validation Loss (MSE) \\\ \n"
        o.write(row)
        o.write("\t\t\\hline\\hline\n")

        id_list = [427, 430, 436, 437, 435, 438, 431, 432, 433, 434]
        for id in id_list:
            o.write(f"\t\t{descr_dict[id]} & {loss_dict[id]} \\\ \n")
            if id == 427:
                o.write("\t\t\\hline\n")
        o.write("\t\t\\hline\n")
        o.write("\t\\end{tabular}\n")
        o.write('''\t\\caption{Neural network ablation results.
        All results in the body of the paper correspond the the Baseline method.
        }\n
        ''')
        o.write("\t\\label{table:ablation}\n")
        o.write("\\end{table}")


if __name__ == '__main__':
    main()
