import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import sys
sys.path.insert(1, 'C:\\Users\\Eric\\OneDrive\\Documents\\Research\\Coding')
from neural_net_utils.base_networks import *
from neural_net_utils.networks import *
import math
import numpy as np
import matplotlib.pyplot as plt
import os

def make_dataset(dir):
    contact_maps = []
    for file in os.listdir(dir):
        contacts_file = dir + '/' + file + '/data_out/contacts.txt'
        contact_maps.append(contacts_file)
        # TODO zero padded??
    return contact_maps

class Sequences2Contacts(Dataset):
    def __init__(self, dir_y, size_y = 1024):
        super(Sequences2Contacts, self).__init__()
        self.dir_y = dir_y
        self.y_paths = sorted(make_dataset(dir_y))
        self.y_transform = transforms.Compose([
            transforms.Resize(size_y), # TODO check interpolation methods
            transforms.ToTensor(),
            # TODO Normalize
        ])

        # TODO x paths
        self.x_transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        y_path = self.y_paths[index]
        y = np.loadtxt(y_path)
        y = self.y_transform(y)

        x_path = self.x_paths[index]
        x = np.loadtxt(x_path)
        x = self.x_transform(x)

        return x, y

    def __len__(self):
        return len(self.y_paths)

    def x2xx(self, x):
        # input x is nxk, output is kxnxn
        n,k = x.shape
        xx = np.zeros((k*2, n, n))
        for i in range(n):
            for j in range(i+1):
                xx[:, i, j] = np.append(x[i], x[j])
                xx[:, j, i] = np.append(x[j], x[i])
        return xx

def getDataloaders(dataset):
    N = len(dataset)
    trainN = math.floor(N * 0.7)
    valN = math.floor(N * 0.2)
    testN = N - trainN - valN
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                        [trainN, valN, testN],
                                        generator = torch.Generator().manual_seed(42))
    # TODO may need to shuffle before split
    train_dataloader = DataLoader(train_dataset, batch_size = 64,
                                    shuffle = True, num_workers = 1)
    val_dataloader = DataLoader(val_dataset, batch_size = 64,
                                    shuffle = True, num_workers = 1)
    test_dataloader = DataLoader(test_dataset, batch_size = 64,
                                    shuffle = True, num_workers = 1)
    # TODO batch_size, num_workers

    return train_dataloader, val_dataloader, test_dataloader

def train(train_loader, val_loader, model, optimizer, criterion, device, save_location,
        epochs, save_mod = 5, print_mod = 100):
    train_loss = []
    val_loss = []
    for e in range(epochs):
        model.train()
        avg_loss = 0
        for t, (x,y) in enumerate(train_loader):
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
                        'optimizer_state_dict': optimizer.state_dict()},
                        save_location)
        if (e + 1) % (epochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        vloss = test(val_loader, model, optimizer, criterion, device)
        train_loss.append(avg_loss/(t+1))
        val_loss.append(vloss)

    return train_loss, val_loss

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


def main(epochs = 100):
    seq2ContactData = Sequences2Contacts('sequences_to_contact_maps/dataset_03_29_21')
    train_dataloader, val_dataloader, test_dataloader = getDataloaders(seq2ContactData)

    model = None

    optimizer = optim.Adam(model.parameters(), lr = 1e-4) # default beta TODO

    train_loss, val_loss = train(train_dataloader, val_dataloader, model, optimizer,
            criterion = F.nll_loss, device = 'cuda:0', save_location = 'model1',
            epochs = epochs)

    plt.plot(np.arange(0, epochs), train_loss, label = 'train_loss')
    plt.plot(np.arange(0, epochs), val_loss, label = 'val_loss')
    plt.legend()
    plt.savefig('train val loss.png')
    plt.close()


if __name__ == '__main__':
    main()
