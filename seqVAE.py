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

def make_dataset(dir):
    contact_maps = []
    for file in os.listdir(dir):
        contacts_file = dir + '/' + file + '/data_out/contacts.txt'
        contact_maps.append(contacts_file)
        # TODO zero padded??
    return contact_maps

class Contacts(Dataset):
    def __init__(self, dir_y, size_y = 1024):
        super(Contacts, self).__init__()
        self.dir_y = dir_y
        self.y_paths = sorted(make_dataset(dir_y))
        self.y_transform = transforms.Compose([
            # TODO resize
            transforms.ToTensor(),
            # TODO Normalize
        ])

    def __getitem__(self, index):
        y_path = self.y_paths[index]
        y = np.loadtxt(y_path)[:1024, :1024] # TODO delete this later
        y = self.y_transform(y)
        return y.float()

    def __len__(self):
        return len(self.y_paths)

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


def main(epochs = 1000, device = 'cuda:0'):
    contactData = Contacts('sequences_to_contact_maps/dataset_03_29_21')
    train_dataloader = DataLoader(contactData, batch_size = 64,
                                    shuffle = True, num_workers = 1)

    model = VAE(1024, 200, 2).to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3) # default beta TODO

    train_loss = trainVAE(train_dataloader, model, optimizer,
            device, save_location = 'model1', epochs = epochs)

    plt.plot(np.arange(0, epochs), train_loss, label = 'train_loss')
    plt.legend()
    plt.savefig('train val loss.png')
    plt.close()

if __name__ == '__main__':
    main()
