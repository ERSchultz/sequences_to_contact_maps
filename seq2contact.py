import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from neural_net_utils.base_networks import *
from neural_net_utils.networks import *
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from utils import *

class Sequences2Contacts(Dataset):
    def __init__(self, dir, size_y = 1024, toxx = False):
        super(Sequences2Contacts, self).__init__()
        self.dir = dir
        self.toxx = toxx
        self.paths = sorted(make_dataset(dir))

    def __getitem__(self, index):
        y_path = self.paths[index] + '/data_out/contacts.txt'
        y = np.loadtxt(y_path)[:1024, :1024] # TODO delete this later
        y = y.reshape(1,1024,1024)

        if self.toxx:
            x_path = self.paths[index] + '/xx.npy'
            x = np.load(x_path)
        else:
            x_path = self.paths[index] + '/x.txt'
            x = np.loadtxt(x_path)

        return torch.Tensor(x), torch.Tensor(y)

    def __len__(self):
        return len(self.paths)

def getDataloaders(dataset, batchSize = 64):
    N = len(dataset)
    trainN = math.floor(N * 0.7)
    valN = math.floor(N * 0.2)
    testN = N - trainN - valN
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                        [trainN, valN, testN],
                                        generator = torch.Generator().manual_seed(42))
    # TODO may need to shuffle before split
    train_dataloader = DataLoader(train_dataset, batch_size = batchSize,
                                    shuffle = True, num_workers = 0)
    val_dataloader = DataLoader(val_dataset, batch_size = batchSize,
                                    shuffle = True, num_workers = 0)
    test_dataloader = DataLoader(test_dataset, batch_size = batchSize,
                                    shuffle = True, num_workers = 0)
    # TODO batch_size, num_workers

    return train_dataloader, val_dataloader, test_dataloader

def train(train_loader, model, optimizer, criterion, device, save_location,
        epochs, save_mod = 5, print_mod = 100):
    train_loss = []
    val_loss = []
    for e in range(epochs):
        model.train()
        avg_loss = 0
        for t, (x,y) in enumerate(train_loader):
            print(t)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            if t % print_mod == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
        if e % save_mod == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': e,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss},
                        save_location)
        if (e + 1) % (epochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        train_loss.append(avg_loss/(t+1))

    return train_loss

def test(loader, model, optimizer, criterion, device):
    loss = 0
    num_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            num_loss += 1
    loss /= num_loss
    print('\nTest set: Avg. loss: {:.4f})\n'.format(loss))
    return loss

def main(dir, epochs = 1000, device = 'cuda:0', k = 2):
    t0 = time.time()
    seq2ContactData = Sequences2Contacts(dir, toxx = True)
    train_dataloader, val_dataloader, test_dataloader = getDataloaders(seq2ContactData)

    if device == 'cuda:0' and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        device = 'cpu'

    model = UNet(nf_in = 2*k, nf_out = 1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-4) # default beta TODO
    criterion = F.mse_loss

    train_loss_arr = train(train_dataloader, model, optimizer,
            criterion, device, save_location = 'model1.pt', epochs = epochs)
    val_loss = test(val_dataloader, model, optimizer, criterion, device)

    print('Total time: {}'.format(time.time() - t0))

    plt.plot(np.arange(0, epochs), train_loss_arr, label = 'train_loss')
    plt.legend()
    plt.savefig('images/train_val_loss.png')
    plt.close()

if __name__ == '__main__':
    clusterdir = '../../../project2/depablo/erschultz/dataset_04_06_21'
    mydir = 'dataset_04_06_21'
    main(clusterdir, 5)
