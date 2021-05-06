import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_net_utils.networks import Akita
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import getBaseParser, finalizeParser, str2list
from neural_net_utils.core_test_train import core_test_train

def argparseSetup():
    parser = getBaseParser()

    # dataloader args
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # model args
    parser.add_argument('--kernel_w_list', type=str2list, default=[5,5], help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=str2list, default=[10,10], help='List of hidden sizes for convolutional layers')
    parser.add_argument('--dilation_list_trunk', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers of trunk')
    parser.add_argument('--bottleneck', type=int, default=10, help='Number of filters in bottleneck (must be <= hidden_size_dilation_trunk)')
    parser.add_argument('--dilation_list_head', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers of head')

    return finalizeParser(parser)

def main():
    opt = argparseSetup()
    print(opt)

    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = False, y_preprocessing   = opt.y_preprocessing,
                                        y_norm = opt.y_norm, x_reshape = True,
                                        crop = opt.crop)

    # Set up model
    model = Akita(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                        opt.dilation_list_trunk,
                        opt.bottleneck,
                        opt.dilation_list_head)
    opt.criterion = F.mse_loss

    core_test_train(seq2ContactData, model, opt)


if __name__ == '__main__':
    main()
