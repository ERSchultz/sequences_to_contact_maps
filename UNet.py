import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_net_utils.networks import *
from neural_net_utils.dataset_classes import *
from neural_net_utils.utils import getDataLoaders
from neural_net_utils.core_test_train import *
import numpy as np
import matplotlib.pyplot as plt
import time

def main(dirname, epochs = 1000, device = 'cuda:0', n = 1024, k = 2):
    t0 = time.time()
    seq2ContactData = Sequences2Contacts(dirname, n = n, k = k, toxx = True)
    train_dataloader, val_dataloader, test_dataloader = getDataloaders(seq2ContactData)

    if device == 'cuda:0' and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        device = 'cpu'

    model = UNet(nf_in = 2, nf_out = 1, nf = 8, out_act = nn.Sigmoid())
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3) # default beta TODO
    criterion = F.mse_loss

    train_loss_arr = train(train_dataloader, model, optimizer,
            criterion, device, save_location = 'models/UNET1.pt', epochs = epochs)
    val_loss = test(val_dataloader, model, optimizer, criterion, device)

    print('Total time: {}'.format(time.time() - t0))
    print('Val loss: {}'.format(val_loss))

    plt.plot(np.arange(0, epochs), train_loss_arr, label = 'train loss')
    plt.axhline(y = val_loss, color = 'black', linestyle = '--', label = 'final val loss')
    plt.legend()
    plt.savefig('images/train_val_loss.png')
    plt.close()

if __name__ == '__main__':
    clusterdir = '../../../project2/depablo/erschultz/dataset_04_06_21'
    mydir = 'dataset_04_06_21'
    main(clusterdir, 20)
