import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(1, 'C:\\Users\\Eric\\OneDrive\\Documents\\Research\\Coding')
from neural_net_utils.base_networks import *

class UNet(nn.Module):
    '''U Net adapted from https://github.com/phillipi/pix2pix.'''
    def __init__(self, nf = 64, nf_in = 3, nf_out = 3, num_downs = 7,
            std_norm = 'batch', std_drop = False, std_drop_p = 0.5):
        # nf is the number of filters input to the final layer (and output from the first layer)
        # num_downs is the number of downsamplings
        # max filters = nf * 8
        super(UNet, self).__init__()
        ub = UnetBlock(nf * 8, nf * 8, norm = std_norm, innermost = True)
        # Reminder that if output_size = None, output_size = input_size
        for _ in range(num_downs - 5):
            ub = UnetBlock(nf * 8, nf * 8, subBlock = ub, norm = std_norm,
                           dropout = std_drop, dropout_p = std_drop_p)
        ub = UnetBlock(nf * 4, nf * 8, subBlock = ub, norm = std_norm)
        ub = UnetBlock(nf * 2, nf * 4, subBlock = ub, norm = std_norm)
        ub = UnetBlock(nf, nf * 2, subBlock = ub, norm = std_norm,)
        sequence = [UnetBlock(nf_in, nf, nf_out, subBlock = ub, norm = std_norm, outermost = True)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PatchGANDiscriminator(nn.Module):
    '''PatchGAN discriminator adapted from https://github.com/phillipi/pix2pix.'''
    def __init__(self, nf = 64, std_norm = 'batch'):
        # nf is the number of filters in the output of the first layer
        super(PatchGANDiscriminator, self).__init__()
        std_kernel = 4
        std_pad = 1
        std_act = 'prelu'

        sequence = [ConvBlock(6, nf, std_kernel, 2, std_pad,
                              bias = True, activation = std_act, norm = None)]
        sequence.append(ConvBlock(nf, nf * 2, std_kernel, 2, std_pad,
                              bias = False, activation = std_act, norm = std_norm))
        sequence.append(ConvBlock(nf * 2, nf * 4, std_kernel, 2, std_pad,
                            bias = False, activation = std_act, norm = std_norm))

        sequence.append(ConvBlock(nf * 4, nf * 8, std_kernel, 1, std_pad,
                              bias = False, activation = std_act, norm = std_norm))

        sequence.append(ConvBlock(nf * 8, 1, std_kernel, 1, std_pad,
                              bias = True, activation = None, norm = None))

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

    def set_requires_grad(self, val):
        for param in self.parameters():
            param.requires_grad = val

class VAE(nn.Module):
    '''Adapted from https://doi.org/10.1371/journal.pcbi.1008262'''
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, input_size)

    def encode(self, input):
        hidden = F.relu(self.fc1(input))
        return self.fc2(hidden), self.fc3(hidden)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent):
        hidden = F.relu(self.fc4(latent))
        return torch.sigmoid(self.fc5(hidden))

    def forward(self, input):
        mu, log_var = self.encode(input)
        latent = self.reparameterize(mu, log_var)
        output = self.decode(latent)
        return output, mu, log_var

    def loss(self, input):
        input = input.view(-1, self.input_size)
        output, mu, log_var = self.forward(input)

        # compute reconstruction loss and kl divergence
        reconst_loss = F.binary_cross_entropy(output, input, reduction = 'sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconst_loss + kl_div
        return loss
