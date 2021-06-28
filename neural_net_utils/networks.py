import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.insert(0, dname)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from base_networks import *
import time

class UNet(nn.Module):
    '''U Net adapted from https://github.com/phillipi/pix2pix.'''
    def __init__(self, nf = 64, nf_in = 3, nf_out = 3, num_downs = 7,
            std_norm = 'batch', std_drop = False, std_drop_p = 0.5, out_act = nn.Tanh()):
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
        ub = UnetBlock(nf, nf * 2, subBlock = ub, norm = std_norm)
        sequence = [UnetBlock(nf_in, nf, nf_out, subBlock = ub, norm = std_norm, outermost = True, out_act = out_act)]
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

class DeepC(nn.Module):
    '''Roughly based on https://doi.org/10.1038/s41592-020-0960-3 (Deepc).'''
    def __init__(self, n, k, kernel_w_list, hidden_sizes_list,
                dilation_list, std_norm, out_act):
        """
        Inputs:
            n: number of particles
            k: number of epigenetic marks
            kernel_w_list: list of kernel widths of convolutional layers
            hidden_sizes_list: list of hidden sizes for convolutional layers
            dilation_list: list of dilations for dilated convolutional layers
            std_norm: default normalization method during training (str)
            out_Act: activation of finally fully connected layer (str or nn.Module)
        """
        super(DeepC, self).__init__()
        model = []

        # Convolution
        input_size = k
        assert len(kernel_w_list) == len(hidden_sizes_list), "length of kernel_w_list ({}) and hidden_sizes_list ({}) must match".format(len(kernel_w_list), len(hidden_sizes_list))
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            model.append(ConvBlock(input_size, output_size, kernel_w, padding = kernel_w//2,
                                    activation = 'prelu', norm = std_norm, dropout = 'drop', dropout_p = 0.2, conv1d = True))
            input_size = output_size

        # Dilated Convolution
        for dilation in dilation_list:
            model.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = 'gated', norm = std_norm, dilation = dilation, residual = True, conv1d = True))

        self.model = nn.Sequential(*model)

        if out_act is None:
            self.out_act = nn.Identity()
        elif issubclass(type(out_act), nn.Module):
            self.out_act = out_act
        elif isinstance(out_act, str):
            if out_act.lower() == 'sigmoid':
                self.out_act = nn.Sigmoid()
            elif out_act.lower() == 'relu':
                self.out_act = nn.ReLU(True)
            else:
                raise Exception("Unkown activation {}".format(out_act))
        else:
            raise Exception("Unknown out_act {}".format(out_act))

    def forward(self, input):
        out = self.model(input)
        out = torch.einsum('bkn, bkm->bnm', out, out)
        out = torch.unsqueeze(out, 1)
        out = self.out_act(out)
        return out

class Akita(nn.Module):
    '''Roughly based on https://doi.org/10.1038/s41592-020-0958-x (Akita).'''
    def __init__(self, n, k, kernel_w_list, hidden_sizes_list,
                dilation_list_trunk,
                bottleneck_size,
                dilation_list_head,
                out_act,
                out_channels,
                std_norm,
                downsample_method, pool_kernel_size = 2, stride_size = 2):
        """
        Inputs:
            n: number of particles
            k: number of epigenetic marks
            kernel_w_list: list of kernel widths of convolutional layers
            hidden_sizes_list: list of hidden sizes for convolutional layers
            dilation_list_trunk: list of dilations for dilated convolutional layers of trunk
            bottleneck_size: size of bottleneck layer
            dilation_list_head: list of dilations for dilated convolutional layers of head
            out_act: activation of finally layer (str)
            out_channels: number of channels in output
            std_norm: default normalization method during training (str)
            downsample_method: method for downsampling prior to AverageTo2d (str)
        """
        super(Akita, self).__init__()

        ## Trunk ##
        trunk = []
        # Convolution
        input_size = k
        assert len(kernel_w_list) == len(hidden_sizes_list), "length of kernel_w_list ({}) and hidden_sizes_list ({}) must match".format(len(kernel_w_list), len(hidden_sizes_list))
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            trunk.append(ConvBlock(input_size, output_size, kernel_w, padding = kernel_w//2,
                                    activation = 'relu', norm = std_norm, conv1d = True))
            input_size = output_size


        # Diaated Convolution
        for dilation in dilation_list_trunk:
            trunk.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = 'relu', norm = std_norm, dilation = dilation, residual = True, conv1d = True))

        # Bottleneck
        trunk.append(ConvBlock(input_size, bottleneck_size, 1, padding = 0, activation = 'relu', conv1d = True))

        # Downsampling
        downsampling_factor = 2
        if downsample_method == 'maxpool':
            trunk.append(nn.MaxPool1d(pool_kernel_size))
        elif downsample_method == 'avgpool':
            trunk.append(nn.AvgPool1d(pool_kernel_size))
        elif downsample_method == 'conv':
            trunk.append(ConvBlock(bottleneck_size, bottleneck_size, stride = stride_size,
                                    activation = 'relu', norm = std_norm, conv1d = True))
        else:
            downsampling_factor = 1
            assert downsample_method is None, "{}".format(downsample_method)

        self.trunk = nn.Sequential(*trunk)

        ## Head ##
        head = []
        head.append(AverageTo2d(True, n // downsampling_factor))
        input_size = bottleneck_size + 1
        head.append(ConvBlock(input_size, input_size, 1, padding = 0, activation = 'relu'))

        # Dilated Convolution
        for dilation in dilation_list_head:
            head.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = 'relu', norm = std_norm, dilation = dilation, residual = True))
            head.append(Symmetrize2D())

        # UpSampling
        if downsample_method is not None:
            head.append(DeconvBlock(input_size, input_size, stride = stride_size,
                                        activation = 'relu', norm = std_norm, output_padding = 1))

        self.head = nn.Sequential(*head)

        # Conversion to out_channels
        self.conv = ConvBlock(input_size, out_channels, 1, padding = 0,
                                activation = out_act)


    def forward(self, input):
        out = self.trunk(input)
        out = self.head(out)
        out = self.conv(out)

        return out

class SimpleEpiNet(nn.Module):
    def __init__(self, n, k, kernel_w_list, hidden_sizes_list, out_act = nn.Sigmoid()):
        super(SimpleEpiNet, self).__init__()
        model = []

        # Convolution
        assert len(kernel_w_list) == len(hidden_sizes_list), "length of kernel_w_list ({}) and hidden_sizes_list ({}) must match".format(len(kernel_w_list), len(hidden_sizes_list))
        input_size = k
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            model.append(ConvBlock(input_size, output_size, kernel_w, padding = kernel_w//2, conv1d = True))
            input_size = output_size

        self.model = nn.Sequential(*model)

        self.act = out_act

    def forward(self, input):
        out = self.model(input)

        out = torch.einsum('...kn, ...km->...nm', out, out)
        out = self.act(out)
        out = torch.unsqueeze(out, 1)
        return out

class GNNAutoencoder(nn.Module):
    def __init__(self, n, input_size, hidden_sizes_list, out_act,
                message_passing, head_architecture, MLP_hidden_sizes_list):
        super(GNNAutoencoder, self).__init__()
        self.n = n
        hidden_size = hidden_sizes_list[0]
        output_size = hidden_sizes_list[1]
        self.output_size = output_size
        if message_passing == 'GCN':
            self.conv1 = torch_geometric.nn.GCNConv(input_size, hidden_size, add_self_loops = False)
            self.conv2 = torch_geometric.nn.GCNConv(hidden_size, output_size, add_self_loops = False)
        else:
            raise Exception("Unkown message_passing {}".format(message_passing))

        self.head_architecture = head_architecture
        if self.head_architecture == 'MLP':
            encode_model = []
            decode_model = []
            input_size = self.n * self.output_size
            for output_size in MLP_hidden_sizes_list:
                encode_model.append(LinearBlock(input_size, output_size, activation = 'relu'))
                decode_model.append(LinearBlock(output_size, input_size, activation = 'relu'))
                input_size = output_size

            self.encode = nn.Sequential(*encode_model)
            decode_model.reverse()
            self.decode = nn.Sequential(*decode_model)

        if out_act is None:
            self.out_act = nn.Identity()
        elif issubclass(type(out_act), nn.Module):
            self.out_act = out_act
        elif isinstance(out_act, str):
            if out_act.lower() == 'sigmoid':
                self.out_act = nn.Sigmoid()
            elif out_act.lower() == 'relu':
                self.out_act = nn.ReLU(True)
            else:
                raise Exception("Unkown activation {}".format(out_act))
        else:
            raise Exception("Unknown out_act {}".format(out_act))

    def forward(self, graph):
        x, edge_index, edge_attr  = graph.x, graph.edge_index, graph.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)

        if self.head_architecture == 'xxT':
            latent = torch.reshape(x, (-1, self.n, self.output_size))
            out = self.out_act(torch.einsum('bij, bkj->bik', latent, latent))
        elif self.head_architecture == 'MLP':
            x = torch.reshape(x, (-1, self.n * self.output_size))
            latent = self.encode(x)
            x = self.decode(latent)
            x = torch.reshape(x, (-1, self.n, self.output_size))
            out = self.out_act(torch.einsum('bij, bkj->bik', x, x))
        else:
            raise Exception("Unkown head architecture {}".format(self.head))

        return out

class FullyConnectedAutoencoder(nn.Module):
    '''
    Fully connected symmetric autoendocer with user defined number of layers and hidden dimension.

    The final value in the hidden_sizes_list is the size of the latent space.

    Note that there is no parameter sharing between the encoder and the decoder.
    '''
    def __init__(self, input_size, hidden_sizes_list):
        '''
        Inputs:
            input_size: size of input (also used as size of output)
            hidden_sizes_list: list of hidden sizes
        '''
        super(FullyConnectedAutoencoder, self).__init__()
        encode_model = []
        decode_model = []
        self.input_size = input_size
        for output_size in hidden_sizes_list:
            encode_model.append(LinearBlock(input_size, output_size, activation = 'relu'))
            decode_model.append(LinearBlock(output_size, input_size, activation = 'relu'))
            input_size = output_size

        self.encode = nn.Sequential(*encode_model)
        decode_model.reverse()
        self.decode = nn.Sequential(*decode_model)


    def forward(self, input):
        # assume input is n x k
        input = torch.reshape(input, (-1, self.input_size))
        latent = self.encode(input)
        output = self.decode(latent)
        return output

class ContactGNN(nn.Module):
    '''
    Graph neural network that maps contact map data to node embeddings of arbitrary length.

    Primary use is to map contact data (formatted as graph) to particle type vector
    where particle type vector is not given as node feature in graph.
    '''
    def __init__(self, n, input_size, hidden_sizes_list, out_act,
                message_passing):
        '''
        Inputs:
            n: number of nodes
            input_size: size of input node feature vector
            hidden_sizes_list: list of node feature vector hidden sizes (final value is output size)
            out_act: output activation
            message_passing: type of message passing algorithm to use
        '''
        super(ContactGNN, self).__init__()
        self.n = n
        if len(hidden_sizes_list) > 2:
            print('Warning: only using first two hidden sizes')
        hidden_size = hidden_sizes_list[0]
        output_size = hidden_sizes_list[1]
        self.output_size = output_size
        if message_passing == 'GCN':
            self.conv1 = torch_geometric.nn.GCNConv(input_size, hidden_size, add_self_loops = False)
            self.conv2 = torch_geometric.nn.GCNConv(hidden_size, output_size, add_self_loops = False)
        else:
            raise Exception("Unkown message_passing {}".format(message_passing))

        if out_act is None:
            self.out_act = nn.Identity()
        elif issubclass(type(out_act), nn.Module):
            self.out_act = out_act
        elif isinstance(out_act, str):
            if out_act.lower() == 'sigmoid':
                self.out_act = nn.Sigmoid()
            elif out_act.lower() == 'relu':
                self.out_act = nn.ReLU(True)
            else:
                raise Exception("Unkown activation {}".format(out_act))
        else:
            raise Exception("Unknown out_act {}".format(out_act))

    def forward(self, graph):
        x, edge_index, edge_attr  = graph.x, graph.edge_index, graph.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        out = self.out_act(x)
        return out

def main():
    f = ContactGNN(1024, 2, [16, 2], None, 'GCN')
    x = f(torch.ones(1024))
    print(x)

if __name__ == '__main__':
    main()
