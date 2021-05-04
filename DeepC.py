import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_net_utils.networks import DeepC
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import getBaseParser, configureOptForCuda, str2list
from neural_net_utils.core_test_train import core_test_train

def argparseSetup():
    parser = getBaseParser()

    # dataloader args
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # model args
    parser.add_argument('--kernel_w_list', type=str2list, default=[5,5], help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=str2list, default=[10,10], help='List of hidden sizes for convolutional layers')
    parser.add_argument('--dilation_list', type=str2list, default=[1,2,4], help='List of dilations for dilated convolutional layers')
    parser.add_argument('--hidden_size_dilation', type=int, default=10, help='Hidden size of dilated convolutional layers')

    opt = parser.parse_args()
    return configureOptForCuda(opt)

def main():
    opt = argparseSetup()
    print(opt)

    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = False, x_reshape = True,
                                        y_norm = opt.y_norm, crop = opt.crop)

    # Set up model
    model = DeepC(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                        opt.dilation_list, opt.hidden_size_dilation)
    criterion = F.mse_loss

    core_test_train(seq2ContactData, model, criterion, opt)


if __name__ == '__main__':
    main()
