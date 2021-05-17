import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_net_utils.networks import Akita
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import argparseSetup, str2list
from neural_net_utils.core_test_train import core_test_train

def main():
    opt = argparseSetup()
    opt.mode = 'debugging'
    if opt.mode == 'debugging':
        # architecture
        opt.k=2
        opt.n=1024
        opt.y_preprocessing='diag'
        opt.y_norm='batch'
        opt.kernel_w_list=str2list('5-5-5')
        opt.hidden_sizes_list=str2list('32-64-96')
        opt.dilation_list=str2list('2-4-8-16')
        opt.bottleneck=32
        opt.dilation_list_head=str2list('2-4-8-16')
        opt.out_act=nn.ReLU(True)

        # hyperparameters
        opt.n_epochs=1
        opt.lr=1e-2
        opt.batch_size=4
        opt.numWorkers=8
        opt.milestones=str2list('1')
        opt.gamma=0.1

        # other
        opt.plot = True
        opt.verbose = True
        opt.data_folder = "/../../../project2/depablo/erschultz/dataset_04_18_21"
        opt.ofile = 'model'
    print(opt)

    # Set up model
    model = Akita(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                        opt.dilation_list_trunk,
                        opt.bottleneck,
                        opt.dilation_list_head,
                        opt.out_act)
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
    else:
        print('Invalid loss: {}'.format(opt.loss))

    core_test_train(model, opt)


if __name__ == '__main__':
    main()
