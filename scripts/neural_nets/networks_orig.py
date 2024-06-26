import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from ..sign_net.sign_net import SignNet
from ..sign_net.transform import to_dense_list_EVD
# from .argparse_utils import finalize_opt, get_base_parser
from .base_networks import (MLP, AverageTo2d, ConvBlock, DeconvBlock,
                            FillDiagonalsFromArray, LinearBlock, Symmetrize2D,
                            UnetBlock, act2module, torch_triu_to_full)
from .pyg_fns import WeightedGATv2Conv, WeightedSignedConv


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
                    opt.log_file, opt.verbose or verbose,
                    opt.use_sign_net, opt.use_sign_plus, opt.k)

        if opt.use_sign_net:
            model = SignNetGNN(opt.node_feature_size, opt.edge_dim, opt.update_hidden_sizes_list[-1],
                            8, 8, opt.k, False, GNN_model,
                            opt.log_file, opt.verbose or verbose)
        elif opt.use_sign_plus:
            model = SignPlus(GNN_model, opt.k)
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
                ofile = sys.stdout, verbose = True,
                sign_net = False, sign_plus = False, k = None):
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
            sign_plus: True if using additional_x from sign_plus
            k: size for sign_plus
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
            output_size = update_hidden_sizes_list[-1]
            self.linear = nn.Linear(input_size+update_hidden_sizes_list[-1], output_size)
            input_size = output_size
        elif sign_plus:
            output_size = update_hidden_sizes_list[-1]
            self.linear = nn.Linear(input_size+k, output_size)
            input_size = output_size
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
                        self.W = self.head[i](latent)
                        out_temp = latent.reshape(-1, self.m, output_size)
                    else:
                        out_temp = self.head[i](latent)

                    if 'asym' in architecture:
                        out_temp = torch.einsum('nik,njk->nij', out_temp @ self.W, out_temp)
                    else:
                        W = self.sym(self.W)
                        if len(W.shape) == 2:
                            left = torch.einsum('nij,jk->nik', out_temp, W)
                        elif len(W.shape) == 3:
                            left = torch.einsum('nij,njk->nik', out_temp, W)
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
