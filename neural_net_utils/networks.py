import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.insert(0, dname)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
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
    def __init__(self, m, k, kernel_w_list, hidden_sizes_list,
                dilation_list, std_norm, act, out_act):
        """
        Inputs:
            m: number of particles
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
                                    activation = act, norm = std_norm, dropout = 'drop', dropout_p = 0.2, conv1d = True))
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
    def __init__(self, m, k, kernel_w_list, hidden_sizes_list,
                dilation_list_trunk,
                bottleneck_size,
                dilation_list_head,
                act,
                out_act,
                out_channels,
                std_norm,
                downsample_method, pool_kernel_size = 2, stride_size = 2):
        """
        Inputs:
            m: number of particles
            k: number of epigenetic marks
            kernel_w_list: list of kernel widths of convolutional layers
            hidden_sizes_list: list of hidden sizes for convolutional layers
            dilation_list_trunk: list of dilations for dilated convolutional layers of trunk
            bottleneck_size: size of bottleneck layer
            dilation_list_head: list of dilations for dilated convolutional layers of head
            act: activation of inner layers
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
                                    activation = act, norm = std_norm, conv1d = True))
            input_size = output_size


        # Diaated Convolution
        for dilation in dilation_list_trunk:
            trunk.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = act, norm = std_norm, dilation = dilation, residual = True, conv1d = True))

        # Bottleneck
        trunk.append(ConvBlock(input_size, bottleneck_size, 1, padding = 0, activation = act, conv1d = True))

        # Downsampling
        downsampling_factor = 2
        if downsample_method == 'maxpool':
            trunk.append(nn.MaxPool1d(pool_kernel_size))
        elif downsample_method == 'avgpool':
            trunk.append(nn.AvgPool1d(pool_kernel_size))
        elif downsample_method == 'conv':
            trunk.append(ConvBlock(bottleneck_size, bottleneck_size, stride = stride_size,
                                    activation = act, norm = std_norm, conv1d = True))
        else:
            downsampling_factor = 1
            assert downsample_method is None, "{}".format(downsample_method)

        self.trunk = nn.Sequential(*trunk)

        ## Head ##
        head = []
        head.append(AverageTo2d(True, m // downsampling_factor))
        input_size = bottleneck_size + 1
        head.append(ConvBlock(input_size, input_size, 1, padding = 0, activation = act))

        # Dilated Convolution
        for dilation in dilation_list_head:
            head.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = act, norm = std_norm, dilation = dilation, residual = True))
            head.append(Symmetrize2D())

        # UpSampling
        if downsample_method is not None:
            head.append(DeconvBlock(input_size, input_size, stride = stride_size,
                                        activation = act, norm = std_norm, output_padding = 1))

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
    def __init__(self, m, k, kernel_w_list, hidden_sizes_list, out_act = nn.Sigmoid()):
        super(SimpleEpiNet, self).__init__()
        model = []

        # Convolution
        assert len(kernel_w_list) == len(hidden_sizes_list), "length of kernel_w_list ({}) and hidden_sizes_list ({}) must match".format(len(kernel_w_list), len(hidden_sizes_list))
        input_size = k
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            model.append(ConvBlock(input_size, output_size, kernel_w, padding = kernel_w//2, conv1d = True))
            input_size = output_size

        self.model = nn.Sequential(*model)

        self.out_act = out_act

    def forward(self, input):
        out = self.model(input)

        out = torch.einsum('...kn, ...km->...nm', out, out)
        out = self.out_act(out)
        out = torch.unsqueeze(out, 1)
        return out

class GNNAutoencoder(nn.Module):
    def __init__(self, m, input_size, hidden_sizes_list, act, head_act, out_act,
                message_passing, head_architecture, head_hidden_sizes_list, parameter_sharing):
        super(GNNAutoencoder, self).__init__()
        self.m = m
        hidden_size = hidden_sizes_list[0]
        output_size = hidden_sizes_list[1]
        self.output_size = output_size
        if message_passing == 'GCN':
            self.conv1 = gnn.GCNConv(input_size, hidden_size, add_self_loops = False)
            self.conv2 = gnn.GCNConv(hidden_size, output_size, add_self_loops = False)
        else:
            raise Exception("Unkown message_passing {}".format(message_passing))

        self.act = actToModule(act)
        self.out_act = actToModule(out_act)

        self.head_architecture = head_architecture
        if self.head_architecture == 'FCAutoencoder':
            input_size = self.m * self.output_size
            self.head = FullyConnectedAutoencoder(input_size, head_hidden_sizes_list, head_act, out_act, parameter_sharing)

    def forward(self, graph):
        x, edge_index, edge_attr  = graph.x, graph.edge_index, graph.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.conv2(x, edge_index, edge_attr)
        # TODO act??

        if self.head_architecture == 'xxT':
            latent = torch.reshape(x, (-1, self.m, self.output_size))
            out = self.out_act(torch.einsum('bij, bkj->bik', latent, latent))
        elif self.head_architecture == 'FCAutoencoder':
            x = torch.reshape(x, (-1, self.m, self.output_size))
            out = self.head(x)
            out = self.out_act(torch.einsum('bij, bkj->bik', x, x))
        else:
            raise Exception("Unkown head architecture {}".format(self.head))

        return out

    def get_latent(self, graph):
        x, edge_index, edge_attr  = graph.x, graph.edge_index, graph.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.conv2(x, edge_index, edge_attr)

        assert self.head_architecture == 'xxT', 'get_latent not supported for {}'.format(self.head_architecture)
        latent = torch.reshape(x, (-1, self.m, self.output_size))
        return latent

class FullyConnectedAutoencoder(nn.Module):
    '''
    Fully connected symmetric autoencoder with user defined number of layers and hidden dimension.

    The final value in the hidden_sizes_list is the size of the latent space.

    Note that there is no parameter sharing between bias of the encoder and the decoder.
    '''
    def __init__(self, input_size, hidden_sizes_list, act, out_act, parameter_sharing):
        '''
        Inputs:
            input_size: size of input (also used as size of output)
            hidden_sizes_list: list of hidden sizes
            act: default activation
            out_act: output activation
            parameter_sharing: True to share parameters between encoder and decoder
        '''
        super(FullyConnectedAutoencoder, self).__init__()
        self.act = actToModule(act)
        self.out_act = actToModule(out_act)

        self.encode_weights = nn.ParameterList()
        self.encode_biases = nn.ParameterList()
        self.decode_weights = nn.ParameterList()
        self.decode_biases = nn.ParameterList()
        self.input_size = input_size
        for output_size in hidden_sizes_list:
            self.encode_weights.append(nn.Parameter(torch.randn(output_size, input_size)))
            self.encode_biases.append(nn.Parameter(torch.randn(output_size)))

            self.decode_biases.append(nn.Parameter(torch.randn(input_size)))
            if not parameter_sharing:
                self.decode_weights.append(nn.Parameter(torch.randn(input_size, output_size)))
            input_size = output_size

    def forward(self, input):
        # assume input is m x k
        N, m, k = input.shape

        input = torch.reshape(input, (N, self.input_size))
        for weight, bias in zip(self.encode_weights, self.encode_biases):
            input = F.linear(input, weight, bias)
            input = self.act(input)

        latent = input

        bias_list = reversed(list(self.decode_biases))
        if self.decode_weights:
            # True if not empty
            for weight, bias in zip(reversed(list(self.decode_weights)), bias_list):
                latent = F.linear(latent, weight, bias)
                latent = self.act(latent)
        else:
            for weight, bias in zip(reversed(list(self.encode_weights)), bias_list):
                latent = F.linear(latent, weight.t(), bias)
                latent = self.act(latent)

        output = self.out_act(latent)
        output = torch.reshape(latent, (N, m, k))
        return output

    def get_latent(self, input):
        # assume input is m x k
        N, m, k = input.shape

        input = torch.reshape(input, (N, self.input_size))
        for weight, bias in zip(self.encode_weights, self.encode_biases):
            input = F.linear(input, weight, bias)
            input = self.act(input)

        latent = input
        return latent

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, m, input_size, hidden_sizes_list, act, out_act, pooling = 'maxpool', conv1d = True):
        '''
        Inputs:
            m: number of particles
            input_size: size of input (also used as size of output)
            hidden_sizes_list: list of hidden sizes, last size is used as latent size of FC layer
            act: default activation
            out_act: output activation
            pooling: method for pooling ('maxpool' or 'strided')
        '''
        super(ConvolutionalAutoencoder, self).__init__()
        encode_model = []
        decode_model = []
        latent_size = hidden_sizes_list.pop()
        print(hidden_sizes_list)
        first = True
        for output_size in hidden_sizes_list:
            encode_model.append(ConvBlock(input_size, output_size, activation = act, pool = pooling, conv1d = conv1d))
            if first:
                decode_model.append(DeconvBlock(output_size, input_size, activation = out_act, conv1d = conv1d, stride = 2, output_padding = 1))
                first = False
            else:
                decode_model.append(DeconvBlock(output_size, input_size, activation = act, conv1d = conv1d, stride = 2, output_padding = 1))
            input_size = output_size

        self.encode = nn.Sequential(*encode_model)
        self.decode = nn.Sequential(*reversed(decode_model))

        input_size = int(m / 2**len(hidden_sizes_list) * input_size)
        print(input_size)
        self.fc1 = LinearBlock(input_size, latent_size, activation = act)
        self.fc2 = LinearBlock(latent_size, input_size, activation = act)

    def forward(self, x):
        x = self.encode(x)
        N, d, m = x.shape
        x = torch.reshape(x, (N, -1))
        latent = self.fc1(x)

        out = self.fc2(latent)
        out = torch.reshape(out, (N, d, m))
        out = self.decode(out)

        return out

    def get_latent(self, x):
        x = self.encode(x)
        N, d, m = x.shape
        x = torch.reshape(x, (N, -1))
        latent = self.fc1(x)

        return latent

class ContactGNN(nn.Module):
    '''
    Graph neural network that maps contact map data to node embeddings of arbitrary length.

    Primary use is to map contact data (formatted as graph) to particle type vector
    where particle type vector is not given as node feature in graph.
    '''
    def __init__(self, m, input_size, hidden_sizes_list, act, inner_act, out_act,
                message_passing, use_edge_weights,
                head_architecture, head_hidden_sizes_list, head_act, use_bias):
        '''
        Inputs:
            m: number of nodes
            input_size: size of input node feature vector
            hidden_sizes_list: list of node feature vector hidden sizes (final value is output size)
            out_act: output activation
            message_passing: type of message passing algorithm to use
            use_edge_weights: True to use edge weights
            head_architecture: type of head architecture
            head_hidden_sizes_list: hidden sizes of head architecture
            use_bias: true to use bias term in message passing (used in head regardless)
        '''
        super(ContactGNN, self).__init__()

        self.m = m
        self.message_passing = message_passing.lower()
        self.use_edge_weights = use_edge_weights
        assert head_architecture is None or head_architecture.lower() in {'fc', 'gcn', 'avg', 'concat', 'outer', 'fc-outer'}, 'Unsupported head architecture {}'.format(head_architecture)
        if head_architecture is not None:
            head_architecture = head_architecture.lower()
        self.head_architecture = head_architecture

        self.act = actToModule(act)
        self.inner_act = actToModule(inner_act, none_mode = True) # added this, non_mode should prevent older models from breaking when reloading
        self.out_act = actToModule(out_act)
        self.head_act = actToModule(head_act)

        model = []
        first_layer = True
        if self.message_passing == 'gcn':
            if self.use_edge_weights:
                fn_header = 'x, edge_index, edge_attr -> x'
            else:
                fn_header = 'x, edge_index -> x'

            for i, output_size in enumerate(hidden_sizes_list):
                module = (gnn.GCNConv(input_size, output_size, bias = use_bias),
                            fn_header)

                if i == len(hidden_sizes_list) - 1:
                    model.extend([module, self.out_act])
                else:
                    if first_layer:
                        self.first_layer = module
                        first_layer = False
                    model.extend([module, self.act])
                input_size = output_size

            self.model = gnn.Sequential('x, edge_index, edge_attr', model)
        elif self.message_passing == 'signedconv':
            assert not self.use_edge_weights and self.head_architecture is not None
            first_layer = True

            for output_size in hidden_sizes_list:
                module = (gnn.SignedConv(input_size, output_size, first_aggr = first_layer, bias = use_bias),
                            'x, pos_edge_index, neg_edge_index -> x')
                model.extend([module, self.act])
                first_layer = False
                input_size = output_size
            input_size *= 2
            # SignedConv convention that output_size is the size of the
            # negative representation and positive representation respectively,
            # so the total length is 2 * output_size

            self.model = gnn.Sequential('x, pos_edge_index, neg_edge_index', model)
        else:
            raise Exception("Unkown message_passing {}".format(message_passing))
        print(self.model)

        ### Head Architecture ###
        head = []
        if self.head_architecture == 'fc-outer':
            # primarily for testing
            self.fc = LinearBlock(input_size, 2, activation = 'sigmoid')
            self.to2D = AverageTo2d(mode = 'outer')
            input_size = 4 # outer squares size
            for i, output_size in enumerate(head_hidden_sizes_list):
                if i == len(hidden_sizes_list) - 1:
                    act = self.out_act
                else:
                    act = self.head_act
                head.append(LinearBlock(input_size, output_size, activation = act))
                input_size = output_size

            self.head = nn.Sequential(*head)
        if self.head_architecture == 'fc':
            for i, output_size in enumerate(head_hidden_sizes_list):
                if i == len(hidden_sizes_list) - 1:
                    act = self.out_act
                    assert output_size == 1, "Final size must be 1 not {}".format(output_size)
                else:
                    act = self.head_act
                head.append(LinearBlock(input_size, output_size, activation = act))
                input_size = output_size

            self.head = nn.Sequential(*head)
        elif self.head_architecture == 'gcn':
            # TODO not sure if I ever tested this
            for i, output_size in enumerate(head_hidden_sizes_list):
                module = (gnn.GCNConv(input_size, output_size),
                            'x, edge_index -> x')
                if i == len(hidden_sizes_list) - 1:
                    head.extend([module, self.out_act])
                else:
                    head.extend([module, self.head_act])
                input_size = output_size

            self.head = gnn.Sequential('x, edge_index', head)
        elif self.head_architecture in {'avg', 'concat', 'outer'}:
            # Uses linear layers according to head_hidden_sizes_list after averaging to 2D
            self.to2D = AverageTo2d(mode = self.head_architecture)
            if self.head_architecture == 'concat':
                input_size *= 2 # concat doubles size
            if self.head_architecture == 'outer':
                input_size *= input_size # outer squares size
            for i, output_size in enumerate(head_hidden_sizes_list):
                if i == len(hidden_sizes_list) - 1:
                    act = self.out_act
                else:
                    act = self.head_act
                head.append(LinearBlock(input_size, output_size, activation = act))
                input_size = output_size

            self.head = nn.Sequential(*head)

    def forward(self, graph):
        if self.message_passing == 'gcn':
            latent = self.model(graph.x, graph.edge_index, graph.edge_attr)
        elif self.message_passing == 'signedconv':
            latent = self.model(graph.x, graph.edge_index, graph.neg_edge_index)

        if self.inner_act is not None:
            latent = self.inner_act(latent)

        if self.head_architecture is None:
            out = latent
        elif self.head_architecture == 'gcn':
            out = self.head(latent, graph.edge_index)
        elif self.head_architecture == 'fc':
            out = self.head(latent)
        elif self.head_architecture in {'avg', 'concat', 'outer'}:
            _, output_size = latent.shape
            latent = latent.reshape(-1, self.m, output_size)
            latent = latent.permute(0, 2, 1)
            latent = self.to2D(latent)
            _, output_size, _, _ = latent.shape
            latent = latent = latent.permute(0, 2, 3, 1)
            out = self.head(latent)
            out = torch.reshape(out, (-1, self.m, self.m))
        elif self.head_architecture == 'fc-outer':
            latent = self.fc(latent)
            _, output_size = latent.shape
            latent = latent.reshape(-1, self.m, output_size)
            latent = latent.permute(0, 2, 1)
            latent = self.to2D(latent)
            _, output_size, _, _ = latent.shape
            latent = latent = latent.permute(0, 2, 3, 1)
            out = self.head(latent)
            out = torch.squeeze(out, 3)

        return out

    def get_first_layer(self, graph):
        out = self.first_layer[0](graph.x, graph.edge_index)
        out = self.act(out)
        return out

def testFullyConnectedAutoencoder():
    model = FullyConnectedAutoencoder(12, [2], 'relu', False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    x = torch.ones((1, 4, 3))
    y = model(x)
    loss = criterion(y, x)
    loss.backward()
    optimizer.step()
    tot_pars = 0
    for k,p in model.named_parameters():
        print(k, p.numel())
        tot_pars += p.numel()
    print('Total parameters: {}'.format(tot_pars))

def testGNNAutoencoder():
    model = GNNAutoencoder(1024, 2, [8, 4], 'relu', 'relu', 'sigmoid', 'GCN', 'FCAutoencoder', [200, 25], True)


def main():
    testGNNAutoencoder()


if __name__ == '__main__':
    main()
