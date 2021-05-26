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
        opt.y_preprocessing='diag'
        opt.y_norm='instance'
        opt.nf = 8

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


    if opt.loss == 'cross_entropy':
        assert opt.y_preprocessing == 'prcnt', 'must use percentile preprocessing with cross entropy'
        assert opt.y_norm is None, 'Cannot normalize with cross entropy'
        assert opt.out_act is None, "Cannot use out_act with cross entropy" # activation combined into loss
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
        model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = opt.out_act)
    elif opt.loss == 'mse':
        opt.criterion = F.mse_loss
        model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = opt.out_act)
    else:
        print('Invalid loss: {}'.format(opt.loss))
    core_test_train(model, opt)


if __name__ == '__main__':
    main()
