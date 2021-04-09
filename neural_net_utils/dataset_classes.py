from torch.utils.data import Dataset, DataLoader
from neural_net_utils.utils import *
import torch

class Sequences2Contacts(Dataset):
    def __init__(self, dirname, toxx = False, y_diag_norm = True, y_reshape = True,
                names = False, crop = None):
        super(Sequences2Contacts, self).__init__()
        self.toxx = toxx
        self.y_diag_norm = y_diag_norm
        self.y_reshape = y_reshape
        self.names = names
        self.crop = crop
        self.paths = sorted(make_dataset(dirname))

    def __getitem__(self, index):
        if self.y_diag_norm and self.crop is None:
            y_path = self.paths[index] + '/y_diag_norm.npy'
            y = np.load(y_path)
        else:
            y_path = self.paths[index] + '/y.npy'
            y = np.load(y_path)
        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
            if self.y_diag_norm:
                diagonal_normalize(y)
        if self.y_reshape:
            y = np.expand_dims(y, 0)
        y = y / np.max(y)

        if self.toxx:
            x_path = self.paths[index] + '/xx.npy'
            x = np.load(x_path)
        else:
            x_path = self.paths[index] + '/x.npy'
            x = np.load(x_path)
            x = np.expand_dims(x, 0)

        if self.names:
            return torch.Tensor(x), torch.Tensor(y), self.paths[index]

        else:
            return torch.Tensor(x), torch.Tensor(y)

    def __len__(self):
        return len(self.paths)

class Contacts(Dataset):
    def __init__(self, dirname, n, y_diag_norm = True):
        super(Contacts, self).__init__()
        self.n = n
        self.y_diag_norm = y_diag_norm
        self.paths = sorted(make_dataset(dirname))

    def __getitem__(self, index):
        y_path = self.paths[index] + '/y.npy'
        y = np.load(y_path)
        if self.y_diag_norm:
            y = diagonal_normalize(y.astype(np.float64))
        y = y.reshape(1, self.n, self.n)
        y = y / np.max(y)
        return y

    def __len__(self):
        return len(self.paths)
