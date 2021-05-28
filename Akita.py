import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_net_utils.networks import Akita
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import argparseSetup, str2list
from neural_net_utils.core_test_train import core_test_train

def main():
    opt = argparseSetup()
    # opt.mode = 'debugging'
    if opt.mode == 'debugging':
        # architecture
        opt.k = 2
        opt.crop = [0, 100]
        opt.n = 100
        opt.y_preprocessing='diag'
        opt.y_norm='batch'
        opt.kernel_w_list=str2list('5-5-5')
        opt.hidden_sizes_list=str2list('4-6-8')
        opt.dilation_list_trunk=str2list('2-4-8-16')
        opt.bottleneck=4
        opt.dilation_list_head=str2list('2-4-8-16')
        opt.out_act=nn.ReLU(True)
        # hyperparameters
        opt.n_epochs=1
        opt.lr=1e-3
        opt.batch_size=1
        opt.milestones=str2list('1')
        opt.gamma=0.1

        # other
        opt.plot = True
        opt.plot_predictions = True
        opt.verbose = False
        if opt.cuda:
            opt.data_folder = "/../../../project2/depablo/erschultz/dataset_04_18_21"
        else:
            opt.data_folder = "dataset_04_18_21"
        opt.ofile = 'model'

    model = Akita(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                        opt.dilation_list_trunk,
                        opt.bottleneck,
                        opt.dilation_list_head,
                        opt.out_act,
                        opt.channels,
                        norm = opt.training_norm,
                        pool = opt.pool)

    core_test_train(model, opt)


if __name__ == '__main__':
    main()
