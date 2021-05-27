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

    # Set up model
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
        opt.channels = 1
    elif opt.loss == 'cross_entropy':
        assert opt.y_preprocessing == 'prcnt', 'must use percentile preprocessing with cross entropy'
        assert opt.y_norm is None, 'Cannot normalize with cross entropy'
        assert opt.out_act is None, "Cannot use output activation with cross entropy"
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
        opt.channels = opt.classes
    else:
        print('Invalid loss: {}'.format(opt.loss))

    model = Akita(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                        opt.dilation_list_trunk,
                        opt.bottleneck,
                        opt.dilation_list_head,
                        opt.out_act,
                        opt.channels,
                        norm = opt.training_norm)

    core_test_train(model, opt)


if __name__ == '__main__':
    main()
