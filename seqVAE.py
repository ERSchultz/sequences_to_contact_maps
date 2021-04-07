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
from utils import make_dataset

class Contacts(Dataset):
    def __init__(self, dir, size_y = 1024):
        super(Contacts, self).__init__()
        self.dir = dir
        self.paths = sorted(make_dataset(dir))

    def __getitem__(self, index):
        y_path = self.paths[index]
        y = np.loadtxt(y_path)[:1024, :1024] # TODO delete this later
        return y.astype(float)

    def __len__(self):
        return len(self.paths)

def trainVAE(train_loader, model, optimizer, device, save_location,
        epochs, save_mod = 5, print_mod = 100):
    train_loss = []
    for e in range(epochs):
        model.train()
        avg_loss = 0
        for t, y in enumerate(train_loader):
            y = y.to(device)
            optimizer.zero_grad()
            loss = model.loss(y)
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
        train_loss.append(avg_loss/(t+1))

    return train_loss


def main(dir, epochs = 1000, device = 'cuda:0'):
    contactData = Contacts(dir)
    train_dataloader = DataLoader(contactData, batch_size = 64,
                                    shuffle = True, num_workers = 1)

    if device == 'cuda:0' and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        device = 'cpu'

    model = VAE(1024, 200, 2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3) # default beta TODO

    train_loss = trainVAE(train_dataloader, model, optimizer,
            device, save_location = 'VAE_model1', epochs = epochs)

    plt.plot(np.arange(0, epochs), train_loss, label = 'train_loss')
    plt.legend()
    plt.savefig('train loss.png')
    plt.close()

if __name__ == '__main__':
    clusterdir = '../../../project2/depablo/skyhl/dataset_04_06_21'
    mydir = 'dataset_04_06_21'
    main(mydir, 1)
