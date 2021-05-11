import torch
from torch.utils.data import Dataset, DataLoader
from neural_net_utils.utils import make_dataset
import torch
import numpy as np
import os

class Names(Dataset):
    "Dataset that only returns names of paths"
    def __init__(self, dirname, min_sample = 0):
        super(Names, self).__init__()
        self.paths = sorted(make_dataset(dirname, minSample = min_sample))

    def __getitem__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

    # TODO make this directly iterable


class Sequences2Contacts(Dataset):
    def __init__(self, dirname, toxx, y_preprocessing, y_norm, x_reshape, ydtype,
                y_reshape, crop, names = False, max = False, min_sample = 0):
        super(Sequences2Contacts, self).__init__()
        self.toxx = toxx
        self.y_norm = y_norm
        if self.y_norm == 'batch':
            assert y_preprocessing is not None, "use instance normalization instead"
            min_max = np.load(os.path.join(dirname, "y_{}_min_max.npy".format(y_preprocessing)))
            print("min, max: ", min_max)
            self.ymin = min_max[0]
            self.ymax = min_max[1]
        self.y_preprocessing = y_preprocessing
        self.x_reshape = x_reshape
        self.ydtype = ydtype
        self.y_reshape = y_reshape
        self.crop = crop
        self.names = names
        self.max = max
        self.paths = sorted(make_dataset(dirname, minSample = min_sample))

    def __getitem__(self, index):
        if self.y_preprocessing is None:
            y_path = os.path.join(self.paths[index], 'y.npy')
            y = np.load(y_path)
        elif self.y_preprocessing == 'diag':
            y_path = os.path.join(self.paths[index], 'y_diag.npy')
            y = np.load(y_path)
        elif self.y_preprocessing == 'prcnt':
            y_path = os.path.join(self.paths[index], 'y_prcnt.npy')
            y = np.load(y_path)
        else:
            print("Warning: Unknown preprocessing: {}".format(self.y_preprocessing))
            y_path = os.path.join(self.paths[index], 'y.npy')
            y = np.load(y_path)

        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]

        if self.y_reshape:
            y = np.expand_dims(y, 0)

        if self.y_norm == 'instance':
            y_max = np.max(y)
            y = y / y_max
        elif self.y_norm == 'batch':
            y_max = self.ymax
            y = (y - self.ymin) / (self.ymax - self.ymin)
        else:
            y_max = -1
            # y_max is unneeded for other y_norms
            # this prevents errors if it is requested (Dataloader doesn't accept Nonetypes)

        if self.toxx:
            x_path = os.path.join(self.paths[index], 'xx.npy')
            x = np.load(x_path)
            if self.crop is not None:
                x = x[:, self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
        else:
            x_path = os.path.join(self.paths[index], 'x.npy')
            x = np.load(x_path)
            if self.crop is not None:
                x = x[:, self.crop[0]:self.crop[1]]
            if self.x_reshape:
                # treat x as 1d image with k channels
                x = x.T

        x = torch.tensor(x, dtype = torch.float32)
        y = torch.tensor(y, dtype = self.ydtype)
        result = [x, y]
        if self.names:
            result.append(self.paths[index])
        if self.max:
            result.append(y_max)

        return result

    def __len__(self):
        return len(self.paths)
