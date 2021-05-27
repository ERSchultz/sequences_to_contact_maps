import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_net_utils.networks import UNet
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import argparseSetup, str2list
from neural_net_utils.core_test_train import core_test_train

def main():
    opt = argparseSetup()
    # opt.mode = 'debugging'
    if opt.mode == 'debugging':
        # Preprocessing
        opt.toxx = True
        opt.toxx_mode = 'mean'
        opt.x_reshape = False

        # architecture
        opt.k=2

        opt.n=1024
        opt.y_preprocessing='prcnt'
        opt.y_norm='instance'
        opt.nf = 8
        opt.out_act = 'sigmoid'

        # hyperparameters
        opt.n_epochs=1
        opt.lr=0.1
        opt.batch_size=4
        opt.milestones=str2list('1')
        opt.gamma=0.1

        # other
        opt.verbose = False
        opt.plot = True
        opt.plot_predictions = True
        opt.data_folder = 'dataset_04_18_21'
        opt.save_mod = 1

    model = UNet(nf_in = opt.k, nf_out = opt.channels, nf = opt.nf, out_act = opt.out_act)

    core_test_train(model, opt)


if __name__ == '__main__':
    main()
