import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

# from .argparse_utils import finalize_opt, get_base_parser
from .base_networks import (MLP, AverageTo2d, ConvBlock, DeconvBlock,
                            FillDiagonalsFromArray, LinearBlock, Symmetrize2D,
                            UnetBlock, act2module, torch_triu_to_full)
from .pyg_fns import WeightedGATv2Conv, WeightedSignedConv
from .sign_net.sign_net import SignNet


## model functions ##
def get_model(opt, verbose = True):
    if opt.model_type == 'SimpleEpiNet':
        model = SimpleEpiNet(opt.input_m, opt.k, opt.kernel_w_list, opt.hidden_sizes_list)
    if opt.model_type == 'UNet':
        model = UNet(opt.nf, opt.k, opt.channels, std_norm = opt.training_norm,
                            out_act = opt.out_act)
    elif opt.model_type == 'DeepC':
        model = DeepC(opt.input_m, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list, opt.training_norm, opt.act, opt.out_act)
    elif opt.model_type == 'Akita':
        model = Akita(opt.input_m, opt.k, opt.kernel_w_list, opt.hidden_sizes_list,
                            opt.dilation_list_trunk, opt.bottleneck,
                            opt.dilation_list_head, opt.act, opt.out_act,
                            opt.channels, opt.training_norm, opt.down_sampling)
    elif opt.model_type.startswith('GNNAutoencoder'):
        model = GNNAutoencoder(opt.input_m, opt.node_feature_size, opt.hidden_sizes_list,
                            opt.act, opt.head_act, opt.out_act,
                            opt.message_passing, opt.head_architecture,
                            opt.head_hidden_sizes_list, opt.parameter_sharing)
    elif opt.model_type == 'SequenceFCAutoencoder':
        model = FullyConnectedAutoencoder(opt.m * opt.k, opt.hidden_sizes_list,
                            opt.act, opt.out_act, opt.parameter_sharing)
    elif opt.model_type == 'SequenceConvAutoencoder':
        model = ConvolutionalAutoencoder(opt.input_m, opt.k, opt.hidden_sizes_list,
                            opt.act, opt.out_act, conv1d = True)
    elif opt.model_type.startswith('ContactGNN'):
        if 'sparse' in opt.transforms:
            GNNClass = SparseContactGNN
        else:
            GNNClass = ContactGNN
        if opt.output_mode is None:
            output_dim = 2 # TODO this might not always be true
        elif opt.output_mode.startswith('diag'):
            output_dim = 1
        else:
            output_dim = 2


        GNN_model = GNNClass(opt.input_m, opt.node_feature_size, output_dim,
                    opt.hidden_sizes_list,
                    opt.encoder_hidden_sizes_list, opt.edge_encoder_hidden_sizes_list,
                    opt.update_hidden_sizes_list,
                    opt.act, opt.inner_act, opt.out_act,
                    opt.message_passing, opt.use_edge_weights or opt.use_edge_attr, opt.edge_dim,
                    opt.head_architecture, opt.head_architecture_2, opt.head_hidden_sizes_list,
                    opt.head_act, opt.use_bias, opt.rescale, opt.gated, opt.dropout,
                    opt.training_norm, opt.num_heads, opt.concat_heads,
                    opt.log_file, opt.verbose or verbose, opt.use_sign_net)

        if opt.use_sign_net:
            model = SignNetGNN(opt.node_feature_size, opt.edge_dim, opt.update_hidden_sizes_list[-1],
                            8, 8, opt.k, False, GNN_model,
                            opt.log_file, opt.verbose or verbose)
        else:
            model = GNN_model

    elif opt.model_type == 'MLP':
        model = MLP(opt.input_m, opt.hidden_sizes_list, opt.use_bias, opt.act,
                            opt.out_act, opt.training_norm, opt.dropout,
                            opt.log_file, opt.verbose or verbose)
        # init = torch.zeros((5, 5))
        # torch.nn.init.xavier_normal_(init)
        # model.sym = Symmetrize2D()
        # model.W = nn.Parameter(init)

        size = int(5*(5+1)/2)
        init = torch.ones(size)
        model.sym = torch_triu_to_full
        model.W = nn.Parameter(init)
    else:
        raise Exception('Invalid model type: {}'.format(opt.model_type))

    return model

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
        sequence = [UnetBlock(nf_in, nf, nf_out, subBlock = ub, norm = std_norm,
                                outermost = True, out_act = out_act)]
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
        len_kw = len(kernel_w_list)
        len_h = len(hidden_sizes_list)
        assert len_kw == len_h, f"length of kernel_w_list != hidden_sizes_list: {len_kw} vs {len_h}"
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            model.append(ConvBlock(input_size, output_size, kernel_w, padding = kernel_w//2,
                                    activation = act, norm = std_norm, dropout = True,
                                    dropout_p = 0.2, conv1d = True))
            input_size = output_size

        # Dilated Convolution
        for dilation in dilation_list:
            model.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = 'gated', norm = std_norm,
                                    dilation = dilation, residual = True, conv1d = True))

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
        len_kw = len(kernel_w_list)
        len_h = len(hidden_sizes_list)
        assert len_kw == len_h, f"length of kernel_w_list != hidden_sizes_list: {len_kw} vs {len_h}"
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            trunk.append(ConvBlock(input_size, output_size, kernel_w, padding = kernel_w//2,
                                    activation = act, norm = std_norm, conv1d = True))
            input_size = output_size


        # Diaated Convolution
        for dilation in dilation_list_trunk:
            trunk.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                    activation = act, norm = std_norm,
                                    dilation = dilation, residual = True, conv1d = True))

        # Bottleneck
        trunk.append(ConvBlock(input_size, bottleneck_size, 1, padding = 0,
                                    activation = act, conv1d = True))

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
                                    activation = act, norm = std_norm,
                                    dilation = dilation, residual = True))
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
        len_kw = len(kernel_w_list)
        len_h = len(hidden_sizes_list)
        assert len_kw == len_h, f"length of kernel_w_list != hidden_sizes_list: {len_kw} vs {len_h}"
        input_size = k
        for kernel_w, output_size in zip(kernel_w_list, hidden_sizes_list):
            model.append(ConvBlock(input_size, output_size, kernel_w,
                                    padding = kernel_w//2, conv1d = True))
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

        self.act = act2module(act)
        self.out_act = act2module(out_act)

        self.head_architecture = head_architecture
        if self.head_architecture == 'FCAutoencoder':
            input_size = self.m * self.output_size
            self.head = FullyConnectedAutoencoder(input_size, head_hidden_sizes_list,
                                                head_act, out_act, parameter_sharing)

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

        assert self.head_architecture == 'xxT', f'get_latent unsupported for {self.head_architecture}'
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
        self.act = act2module(act)
        self.out_act = act2module(out_act)

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
    def __init__(self, m, input_size, hidden_sizes_list, act, out_act,
                pooling = 'maxpool', conv1d = True):
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
        first = True
        for output_size in hidden_sizes_list:
            encode_model.append(ConvBlock(input_size, output_size, activation = act,
                                            pool = pooling, conv1d = conv1d))
            if first:
                decode_model.append(DeconvBlock(output_size, input_size,
                                            activation = out_act, conv1d = conv1d,
                                            stride = 2, output_padding = 1))
                first = False
            else:
                decode_model.append(DeconvBlock(output_size, input_size,
                                            activation = act, conv1d = conv1d,
                                            stride = 2, output_padding = 1))
            input_size = output_size

        self.encode = nn.Sequential(*encode_model)
        self.decode = nn.Sequential(*reversed(decode_model))

        input_size = int(m / 2**len(hidden_sizes_list) * input_size)
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
    '''
    def __init__(self, m, input_size, output_dim, MP_hidden_sizes_list,
                node_encoder_hidden_sizes_list, edge_encoder_hidden_sizes_list,
                update_hidden_sizes_list,
                act, inner_act, out_act,
                message_passing, use_edge_attr, edge_dim,
                head_architecture, head_architecture_2, head_hidden_sizes_list,
                head_act, use_bias, rescale, gated, dropout,
                training_norm, num_heads, concat_heads,
                ofile = sys.stdout, verbose = True, sign_net = False):
        '''
        Inputs:
            m: number of nodes
            input_size: size of input node feature vector
            output_dim: number of dimensions in output (size of each dimension is m)
            MP_hidden_sizes_list: list of node feature vector hidden sizes during message passing
            node_encoder_hidden_sizes_list: list of hidden sizes for MLP encoder
            edge_encoder_hidden_sizes_list: list of hidden sizes for MLP encoder
            update_hidden_sizes_list: list of hidden sizes for MLP for update during message passing
            inner_hidden_sizes_list: list of hidden sizes for MLP for final update during message passing
            act (str): activation function
            inner_act (str): inner activation function
            out_act (str): output activation
            message_passing (str): type of message passing algorithm to use
                                    {idendity, gcn, signedconv, z, gat, weighted_gat}
            use_edge_attr: True to use edge attributes/weights
            edge_dim: 0 for edge weights, 1+ for edge_attr
            head_architecture: type of head architecture {None, fc, inner, bilinear, AverageTo2d.mode_options}
            head_architecture_2: type of head architecture {None, fc, inner, bilinear, AverageTo2d.mode_options}
            head_hidden_sizes_list: hidden sizes of head architecture
            head_act: activation for head_architecture (and head_architecture_2)
            use_bias: True to use bias term - applies for message passing and head
            rescale: rescale by factor <rescale> (None to skip) uses F.interpolate
            gated: True to use gated residual connection (https://doi.org/10.3389/fmolb.2021.647915)
            dropout: Value for dropout (0.0 for no dropout)
            training_norm: Normalization layer
            sign_net: True if using additional_x from sign_net
        '''
        super(ContactGNN, self).__init__()

        self.m = m
        self.output_dim = output_dim
        self.message_passing = message_passing.lower()
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim
        if head_architecture is not None:
            head_architecture = head_architecture.lower()
        self.head_architecture = head_architecture
        self.head_architecture_2 = head_architecture_2
        self.to2D = AverageTo2d(mode = None)

        # set up activations
        self.act = act2module(act)
        self.inner_act = act2module(inner_act)
        self.out_act = act2module(out_act)
        self.head_hidden_sizes_list = head_hidden_sizes_list
        if head_hidden_sizes_list is not None and len(head_hidden_sizes_list) > 1:
            self.head_act = act2module(head_act)
            # only want this to show up as a parameter if actually needed
        else:
            self.head_act = None
        self.use_bias = use_bias
        self.rescale = rescale
        self.gated = gated
        self.dropout = dropout

        ### Encoder Architecture ###
        self.node_encoder = None
        self.edge_encoder = None
        if node_encoder_hidden_sizes_list is not None:
            encoder = []
            for output_size in node_encoder_hidden_sizes_list:
                module = gnn.Linear(input_size, output_size, bias = use_bias)
                encoder.extend([(module, 'x -> x'), self.act])
                input_size = output_size
            self.node_encoder = gnn.Sequential('x', encoder)

        if edge_encoder_hidden_sizes_list is not None:
            encoder = []
            input_size += edge_dim
            for output_size in edge_encoder_hidden_sizes_list:
                encoder.append(LinearBlock(input_size, output_size, activation = self.act,
                                        bias = use_bias))
                input_size = output_size
            self.edge_encoder = nn.Sequential(*encoder)

        if sign_net:
            # print(input_size, update_hidden_sizes_list[-1])
            self.linear = nn.Linear(input_size+update_hidden_sizes_list[-1], update_hidden_sizes_list[-1])
            input_size = update_hidden_sizes_list[-1]
        else:
            self.linear = None



        ### Trunk Architecture ###
        model = []
        if self.message_passing == 'identity':
            # debugging option to skip message passing
            self.model = None
        elif self.message_passing in {'gcn', 'transformer', 'gat', 'weighted_gat'}:
            if self.use_edge_attr:
                inputs = 'x, edge_index, edge_attr'
            else:
                inputs = 'x, edge_index'
            fn_header = f'{inputs} -> x'

            for i, output_size in enumerate(MP_hidden_sizes_list):
                if self.message_passing == 'gcn':
                    module = gnn.GCNConv(input_size, output_size, bias = use_bias)
                elif self.message_passing == 'transformer':
                    module = gnn.TransformerConv(input_size, output_size,
                                            heads = num_heads,
                                            edge_dim = self.edge_dim)
                elif self.message_passing == 'gat':
                    module = gnn.GATv2Conv(input_size, output_size,
                                            heads = num_heads, concat = concat_heads,
                                            edge_dim = self.edge_dim,
                                            bias = use_bias)
                elif self.message_passing == 'weighted_gat':
                    module = WeightedGATv2Conv(input_size, output_size,
                                            heads = num_heads, concat = concat_heads,
                                            dropout = self.dropout,
                                            edge_dim = self.edge_dim, edge_dim_MP = True,
                                            bias = use_bias)
                model.append((module, fn_header))
                if concat_heads:
                    input_size = output_size * num_heads
                else:
                    input_size = output_size

                if i == len(MP_hidden_sizes_list) - 1:
                    act = self.inner_act
                else:
                    act = self.act
                if update_hidden_sizes_list is not None:
                    model.append(MLP(input_size, update_hidden_sizes_list, use_bias,
                                    self.act, act, dropout = self.dropout,
                                    dropout_last_layer = True, gated = self.gated))
                    input_size = update_hidden_sizes_list[-1]
                else:
                    model.append(act)

            if training_norm == 'instance':
                model.append(gnn.InstanceNorm(input_size))
            elif training_norm is not None:
                raise Exception(f'Invalid training_norm: {training_norm}')

            self.model = gnn.Sequential('x, edge_index, edge_attr', model)
        elif self.message_passing == 'signedconv':
            if self.use_edge_attr:
                inputs = 'x, pos_edge_index, neg_edge_index, pos_edge_attr, neg_edge_attr'
            else:
                inputs = 'x, pos_edge_index, neg_edge_index'
            fn_header = f'{inputs} -> x'
            assert self.head_architecture is not None
            first_layer = True

            for output_size in MP_hidden_sizes_list:
                model.append((WeightedSignedConv(input_size, output_size,
                                first_aggr = first_layer, bias = use_bias,
                                edge_dim = self.edge_dim),
                                fn_header))
                input_size = output_size
                input_size *= 2
                # SignedConv convention that output_size is the size of the
                # negative representation and positive representation respectively,
                # so the total length is 2 * output_size

                if update_hidden_sizes_list is not None:
                    for update_output_size in update_hidden_sizes_list:
                        model.extend([gnn.Linear(input_size, update_output_size, bias = use_bias), self.act])
                        input_size = update_output_size
                else:
                    model.append(self.act)
                input_size //= 2
                first_layer = False

            # replace final act with inner_act
            model.pop()
            model.append(self.inner_act)

            input_size *= 2
            if training_norm == 'instance':
                model.append(gnn.InstanceNorm(input_size))
            elif training_norm is not None:
                raise Exception(f'Invalid training_norm: {training_norm}')

            self.model = gnn.Sequential(inputs, model)
        elif self.message_passing == 'z':
            # uses prior model to predict particle types
            # designed for debugging
            self.model = None
        else:
            raise Exception("Unkown message_passing {}".format(message_passing))

        self.latent_size = input_size
        # save input_size to latent_size
        # this is the output_size of latent space
        # and the input size for head_architecture

        ### Head Architecture ###
        self.head_1 = self.process_head_architecture(self.head_architecture)
        self.head_2 = self.process_head_architecture(self.head_architecture_2)
        self.head = [self.head_1, self.head_2]

        if verbose:
            print("#### ARCHITECTURE ####", file = ofile)
            print('Node Encoder:\n', self.node_encoder, '\n', file = ofile)
            print('Edge Encoder:\n', self.edge_encoder, '\n', file = ofile)
            print('Linear:\n', self.linear, '\n', file = ofile)
            print('Model:\n', self.model, '\n', file = ofile)
            print('Head 1:\n', self.head_1, '\n', file = ofile)
            print('Head 2:\n', self.head_2, '\n', file = ofile)

    def process_head_architecture(self, head_architecture):
        split = head_architecture.split('_')
        if head_architecture is None:
            head = None
        elif head_architecture.startswith('fc'):
            if len(split) > 1:
                output_size = int(split[1])
                head_hidden_sizes_list = self.head_hidden_sizes_list + [output_size]
            else:
                head_hidden_sizes_list = self.head_hidden_sizes_list
            head_list = []
            input_size = self.latent_size * self.m
            head_list.append(MLP(input_size, head_hidden_sizes_list, self.use_bias,
                                self.head_act, self.out_act, dropout = self.dropout))
            if 'fill' in head_architecture:
                head_list.append(FillDiagonalsFromArray())

            head = nn.Sequential(*head_list)
        elif head_architecture.startswith('bilinear'):
            head = 'Bilinear'
            if 'chi' in split:
                assert 'triu' in split, f"{head_architecture}"
                size = int(self.latent_size*(self.latent_size+1)/2)
                input_size = self.latent_size * self.m
                head_hidden_sizes_list = self.head_hidden_sizes_list + [size]
                head = MLP(input_size, head_hidden_sizes_list, self.use_bias,
                            self.head_act, self.out_act, dropout = self.dropout)


            if 'triu' in split:
                size = int(self.latent_size*(self.latent_size+1)/2)
                init = torch.zeros(size)
                self.sym = torch_triu_to_full
            else:
                init = torch.zeros((self.latent_size, self.latent_size))
                torch.nn.init.xavier_normal_(init)
                self.sym = Symmetrize2D()
            if 'chi' not in split:
                self.W = nn.Parameter(init)
        elif head_architecture == 'inner':
            head = 'Inner'
        elif head_architecture in self.to2D.mode_options:
            # Uses linear layers according to head_hidden_sizes_list after converting to 2D
            self.to2D.mode = head_architecture # change mode
            input_size = self.latent_size
            # determine input_size
            if head_architecture == 'concat':
                input_size *= 2 # concat doubles size
            elif head_architecture == 'outer':
                input_size *= input_size # outer squares size
            elif head_architecture == 'concat-outer':
                input_size = input_size**2 + 2 * input_size
            elif head_architecture == 'avg-outer':
                input_size += input_size**2

            head_list = []
            for i, output_size in enumerate(self.head_hidden_sizes_list):
                if i == len(self.head_hidden_sizes_list) - 1:
                    act = self.out_act
                else:
                    act = self.head_act
                head_list.append(LinearBlock(input_size, output_size, activation = act,
                                        bias = self.use_bias, dropout = self.dropout))
                input_size = output_size

            head = nn.Sequential(*head_list)
        else:
            raise Exception(f"Unkown head_architecture {head_architecture}")

        return head

    def forward(self, graph, additional_x=None):
        latent = self.latent(graph, additional_x)
        _, output_size = latent.shape

        if self.head_architecture is None and self.head_architecture_2 is None:
            return latent

        first = True
        for i, architecture in enumerate([self.head_architecture, self.head_architecture_2]):
            if architecture is None:
                continue
            elif architecture.startswith('fc'):
                latent = latent.reshape(-1, self.m * output_size)
                out_temp = self.head[i](latent)
            else:
                if architecture.startswith('bilinear'):
                    latent = latent.reshape(-1, self.m, output_size)
                    if self.head[i] == 'Bilinear':
                        out_temp = latent
                    elif 'chi' in architecture:
                        latent = latent.reshape(-1, self.m * output_size)
                        W = self.head[i](latent)
                        self.W = W.reshape(-1)
                        out_temp = latent.reshape(-1, self.m, output_size)
                    else:
                        out_temp = self.head[i](latent)

                    if 'asym' in architecture:
                        out_temp = torch.einsum('nik,njk->nij', out_temp @ self.W, out_temp)
                    else:
                        left = torch.einsum('nij,jk->nik', out_temp, self.sym(self.W))
                        out_temp = torch.einsum('nik,njk->nij', left, out_temp)
                elif architecture == 'inner':
                    latent = latent.reshape(-1, self.m, output_size)
                    out_temp = torch.einsum('nik, njk->nij', latent, latent)
                elif architecture in self.to2D.mode_options:
                    latent = latent.reshape(-1, self.m, output_size)
                    latent = latent.permute(0, 2, 1) # permute to combine over m index
                    out_temp = self.to2D(latent)
                    out_temp = out_temp.permute(0, 2, 3, 1) # permute back
                    latent = latent.permute(0, 2, 1) # permute back
                    out_temp = self.head[i](out_temp)
                    if len(out_temp.shape) > 3:
                        out_temp = torch.squeeze(out_temp, 3)

                if self.rescale is not None:
                    out_temp = torch.unsqueeze(out_temp, 1)
                    m_new = int(self.m * self.rescale)
                    out_temp = F.interpolate(out_temp, size = (m_new, m_new))
                    out_temp = torch.squeeze(out_temp, 1)

            if first:
                out = out_temp
                first = False
            else:
                try:
                    out = out + out_temp
                except RuntimeError:
                    print(out.shape, out_temp.shape)
                    raise

        return out

    def latent(self, graph, additional_x):
        if self.node_encoder is not None:
            x = self.node_encoder(graph.x)
        else:
            x = graph.x

        if additional_x is not None:
            x = self.linear(torch.cat([x, additional_x], dim=-1))

        if self.edge_encoder is not None:
            row, col = graph.edge_index
            concat = torch.cat((x[row], x[col], graph.edge_attr), dim = -1)
            edge_attr = self.edge_encoder(concat)
        else:
            edge_attr = graph.edge_attr

        if self.message_passing == 'identity':
            latent = x
        elif self.message_passing in {'gcn', 'transformer', 'gat', 'weighted_gat'}:
            latent = self.model(x, graph.edge_index, edge_attr)
        elif self.message_passing == 'signedconv':
            if self.use_edge_attr:
                latent = self.model(x, graph.pos_edge_index, graph.neg_edge_index,
                                    graph.pos_edge_attr, graph.neg_edge_attr)
            else:
                latent = self.model(x, graph.pos_edge_index, graph.neg_edge_index)

        return latent

    def diagonal_component(self, graph, additional_x=None):
        latent = self.latent(graph, additional_x)

        for i, architecture in enumerate([self.head_architecture, self.head_architecture_2]):
            if architecture is None:
                continue
            elif architecture.startswith('fc'):
                latent_copy = torch.clone(latent)
                latent_copy = latent_copy.reshape(-1)
                return self.head[i](latent_copy)
            elif architecture in self.to2D.mode_options:
                _, output_size = latent.shape
                latent_copy = torch.clone(latent)
                latent_copy = latent_copy.reshape(-1, self.m, output_size)
                latent_copy = latent_copy.permute(0, 2, 1) # permute to combine over m index
                latent_copy = self.to2D(latent_copy)
                _, output_size, _, _ = latent_copy.shape
                latent_copy = latent_copy.permute(0, 2, 3, 1) # permute back
                out_temp = self.head[i](latent_copy)
                if len(out_temp.shape) > 3:
                    out_temp = torch.squeeze(out_temp, 3)
                return out_temp

        return None

    def plaid_component(self, graph, additional_x=None):
        latent = self.latent(graph, additional_x)
        _, output_size = latent.shape

        out_temp = None
        for i, architecture in enumerate([self.head_architecture, self.head_architecture_2]):
            if architecture is None:
                continue
            elif architecture.startswith('bilinear'):
                latent = latent.reshape(-1, self.m, output_size)
                if self.head[i] == 'Bilinear':
                    out_temp = latent
                elif 'chi' in architecture:
                    latent = latent.reshape(-1, self.m * output_size)
                    W = self.head[i](latent)
                    self.W = W.reshape(-1)
                    out_temp = latent.reshape(-1, self.m, output_size)
                else:
                    out_temp = self.head[i](latent)

                if 'asym' in architecture:
                    out_temp = torch.einsum('nik,njk->nij', out_temp @ self.W, out_temp)
                else:
                    left = torch.einsum('nij,jk->nik', out_temp, self.sym(self.W))
                    out_temp = torch.einsum('nik,njk->nij', left, out_temp)
            elif architecture == 'inner':
                latent = latent.reshape(-1, self.m, output_size)
                out_temp = torch.einsum('nik, njk->nij', latent, latent)


        if self.rescale is not None:
            out_temp = torch.unsqueeze(out_temp, 1)
            m_new = int(self.m * self.rescale)
            out_temp = F.interpolate(out_temp, size = (m_new, m_new))
            out_temp = torch.squeeze(out_temp, 1)

        return out_temp

class SignNetGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, nl_signnet,
                    nl_rho, k, ignore_eigval, gnn_model, ofile, verbose):
        super().__init__()
        self.sign_net = SignNet(n_hid, nl_signnet, nl_rho, ignore_eigval, k)
        self.gnn = gnn_model

        if verbose:
            print("#### ARCHITECTURE ####", file = ofile)
            print('Sign Net:\n', self.sign_net, '\n', file = ofile)


    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        pos = self.sign_net(data)
        return self.gnn(data, pos)

    def plaid_component(self, data):
        pos = self.sign_net(data)
        return self.gnn.plaid_component(data, pos)

    def diagonal_component(self, data):
        pos = self.sign_net(data)
        return self.gnn.diagonal_component(data, pos)


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
    model = GNNAutoencoder(1024, 2, [8, 4], 'relu', 'relu', 'sigmoid', 'GCN',
                            'FCAutoencoder', [200, 25], True)


if __name__ == '__main__':
    testGNNAutoencoder()
