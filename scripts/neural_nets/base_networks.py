import sys
import time

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import solve, symbols


def act2module(act, none_mode = False, in_place = True):
    '''
    Converts input activation, act, to nn.Module activation.

    Inputs:
        act: activation function - string, nn.Module, or None
        none_mode: True to return None instead of nn.Identity for no activation

    act can be None, a string, or nn.Module already.
    '''
    if act is None:
        act = nn.Identity()
        if none_mode:
            return None
    elif issubclass(type(act), nn.Module):
        pass
    elif isinstance(act, str):
        if act.lower() == 'none':
            act = nn.Identity()
            if none_mode:
                return None
        elif act.lower() == 'sigmoid':
            act = nn.Sigmoid()
        elif act.lower() == 'relu':
            act = nn.ReLU(in_place)
        elif act.lower() == 'prelu':
            act = nn.PReLU()
        elif act.lower() == 'leaky':
            act = nn.LeakyReLU(0.2, in_place)
        elif act.lower() == 'tanh':
            act = nn.Tanh()
        else:
            raise Exception("Unsupported activation {}".format(act))
    else:
        raise Exception("Unsupported activation {}".format(act))

    return act

def torch_triu_to_full(arr):
    '''Convert array of upper triangle to symmetric matrix.'''
    # infer m given length of upper triangle
    if len(arr.shape) == 1:
        l, = arr.shape
        x, y = symbols('x y')
        y=x*(x+1)/2-l
        result=solve(y)
        m = int(np.max(result))

        # need to reconstruct from upper traingle
        out = torch.zeros((m, m), dtype = torch.float32)
        if arr.is_cuda:
            out = out.to(arr.get_device())
        out[np.triu_indices(m)] = arr
        out = out + torch.triu(out, 1).t()
    elif len(arr.shape) == 2:
        b, l = arr.shape
        x, y = symbols('x y')
        y=x*(x+1)/2-l
        result=solve(y)
        m = int(np.max(result))

        # need to reconstruct from upper traingle
        out = torch.zeros((b, m, m), dtype = torch.float32)
        if arr.is_cuda:
            out = out.to(arr.get_device())
        for i in range(b):
            out[i, :, :][np.triu_indices(m)] = arr[i, :]
        out = out + torch.transpose(torch.triu(out, 1), 1, 2)

    return out

class UnetBlock(nn.Module):
    '''U Net Block adapted from https://github.com/phillipi/pix2pix.'''
    def __init__(self, input_size, inner_size, output_size = None, subBlock = None,
                 outermost = False, innermost = False,
                 kernel_size = 4, stride = 2, padding = 2, bias = True,
                 activation1 = 'leaky', activation2 = 'relu', norm = None,
                 dropout = False, dropout_p = 0.5, out_act = nn.Tanh()):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if output_size is None:
            output_size = input_size

        if norm == 'batch':
            norm1 = nn.BatchNorm2d(inner_size)
            norm2 = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            norm1 = nn.InstanceNorm2d(inner_size)
            norm2 = nn.InstanceNorm2d(output_size)
        elif norm is None:
            norm1 = nn.Identity()
            norm2 = nn.Identity()

        conv1 = nn.Conv2d(input_size, inner_size, kernel_size, stride, padding, bias)

        act1 = act2module(activation1)
        act2 = act2module(activation2)

        if outermost:
            conv2 = nn.ConvTranspose2d(inner_size * 2, output_size, kernel_size, stride, padding)
            down = [conv1]
            if out_act is not None:
                out_act = act2module(out_act)
                up = [act2, conv2, out_act]
            else:
                up = [act2, conv2]
            model = down + [subBlock] + up
        elif innermost:
            conv2 = nn.ConvTranspose2d(inner_size, output_size, kernel_size, stride, padding, bias)
            down = [act1, conv1]
            up = [act2, conv2, norm2]
            model = down + up
        else:
            conv2 = nn.ConvTranspose2d(inner_size * 2, output_size, kernel_size, stride, padding, bias)
            down = [act1, conv1, norm1]
            up = [act2, conv2, norm2]
            if dropout:
                model = down + [subBlock] + up + [nn.Dropout(dropout_p)]
            else:
                model = down + [subBlock] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # skip connection
            return torch.cat([x, self.model(x)], 1)

class ResnetBlock(nn.Module):
    def __init__(self, num_channels, kernel_size = 3, stride = 1, padding = 1, bias = True,
                 activation = 'prelu', norm = None, dropout = None, dropout_p = 0.5):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size,
                                     stride, padding, bias = bias)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size,
                                     stride, padding, bias = bias)

        self.norm = norm
        if self.norm == 'batch':
            self.norm = nn.BatchNorm2d(num_channels)
        elif self.norm == 'instance':
            self.norm = nn.InstanceNorm2d(num_channels)

        self.drop = dropout
        if self.drop == 'drop2d':
            self.drop = nn.Dropout2d(dropout_p)
        elif self.drop == 'drop':
            self.drop = nn.Dropout(dropout_p)

        self.act = act2module(activation)

    def forward(self, x):
        if self.norm is not None:
            out = self.norm(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.act is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.norm(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, x)

        if self.drop is not None:
            out = self.drop(out)

        if self.act is not None:
            out = self.act(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size = 3, stride = 1, padding = 1, bias = True,
                 activation = 'prelu', norm = None, pool = None, pool_kernel_size = 2, dropout = False,
                 dropout_p = 0.5, dilation = 1, residual = False, conv1d = False):
        super(ConvBlock, self).__init__()

        self.residual = residual
        if isinstance(activation, str) and activation.lower() == 'gated':
            self.gated = True
        else:
            self.gated = False

        if conv1d:
            conv_fn = nn.Conv1d
        else:
            conv_fn = nn.Conv2d

        model = [conv_fn(input_size, output_size, kernel_size,
                                    stride, padding, dilation, bias = bias)]
        model2 = [conv_fn(input_size, output_size, kernel_size,
                                    stride, padding, dilation, bias = bias)] # only used if self.gated

        if norm == 'batch':
            if conv1d:
                batch_fn = nn.BatchNorm1d
            else:
                batch_fn = nn.BatchNorm2d
        elif norm == 'instance':
            if conv1d:
                batch_fn = nn.InstanceNorm1d
            else:
                batch_fn = nn.InstanceNorm2d
        else:
            batch_fn = nn.Identity
        model.append(batch_fn(output_size))
        model2.append(batch_fn(output_size))

        if pool == 'maxpool':
            if conv1d:
                maxpool_fn = nn.MaxPool1d
            else:
                maxpool_fn = nn.MaxPool2d
            model.append(maxpool_fn(pool_kernel_size))
            model2.append(maxpool_fn(pool_kernel_size))

        if dropout:
            if conv1d:
                dropout_fn = nn.Dropout
            else:
                dropout_fn = nn.Dropout2d
            model.append(dropout_fn(dropout_p))
            model2.append(dropout_fn(dropout_p))

        if activation is None:
            pass
        elif isinstance(activation, str) and activation.lower() == 'gated':
            model.append(nn.Tanh())
            model2.append(nn.Sigmoid())
            self.model2 = nn.Sequential(*model2)
        else:
            model.append(act2module(activation))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        if self.gated:
            out = torch.mul(out, self.model2(x))

        if self.residual:
            return torch.add(out, x)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size = 3, stride = 1, padding = 1,
                 bias = True, activation = 'prelu', norm = None, dropout = None,
                 dropout_p = 0.5, output_padding = 0, conv1d = False):
        super(DeconvBlock, self).__init__()

        if conv1d:
            conv_fn = nn.ConvTranspose1d
        else:
            conv_fn = nn.ConvTranspose2d

        model = [conv_fn(input_size, output_size, kernel_size,
                                               stride, padding, bias = bias, output_padding = output_padding)]

        if norm == 'batch':
            if conv1d:
                batch_fn = nn.BatchNorm1d
            else:
                batch_fn = nn.BatchNorm2d
        elif norm == 'instance':
            if conv1d:
                batch_fn = nn.InstanceNorm1d
            else:
                batch_fn = nn.InstanceNorm2d
        else:
            batch_fn = nn.Identity
        model.append(batch_fn(output_size))

        if dropout == 'drop':
            if conv1d:
                dropout_fn = nn.Dropout
            else:
                dropout_fn = nn.Dropout2d
            model.append(dropout_fn(dropout_p))
            model2.append(dropout_fn(dropout_p))

        if activation is None:
            pass
        elif isinstance(activation, str) and activation.lower() == 'gated':
            model.append(nn.Tanh())
            model2.append(nn.Sigmoid())
            self.model2 = nn.Sequential(*model2)
        else:
            model.append(act2module(activation))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, bias = True, activation = 'prelu',
                 norm = None, dropout = 0.0):
        super(LinearBlock, self).__init__()

        model =  [nn.Linear(input_size, output_size, bias = bias)]

        if norm == 'batch':
            model.append(nn.BatchNorm1d(output_size))

        if dropout != 0.0:
            model.append(nn.Dropout(dropout))

        if activation is not None:
            model.append(act2module(activation))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, bias = True, act = 'relu',
                out_act = None, norm = None, dropout = 0.0,
                dropout_last_layer = False, gated = False,
                ofile = sys.stdout, verbose = False):
        super(MLP, self).__init__()
        self.gated = gated
        model = []

        input_size = in_channels
        for i, output_size in enumerate(hidden_channels):
            if i == len(hidden_channels) - 1:
                act = out_act
                if not dropout_last_layer:
                    dropout = False # no dropout in last layer
            model.append(LinearBlock(input_size, output_size, activation = act,
                            norm = norm, dropout = dropout))
            input_size = output_size

        self.model = nn.Sequential(*model)

        if self.gated:
            assert in_channels == hidden_channels[-1], f'{in_channels} != {hidden_channels[-1]}'
            self.D = nn.Parameter(torch.randn(2*in_channels))
            self.b = nn.Parameter(torch.zeros((1)))

        if verbose:
            print("#### ARCHITECTURE ####", file = ofile)
            print(self.model, file = ofile)

    def forward(self, input):
        if self.gated:
            x_in = torch.clone(input)
            x_out = self.model(input)

            x_all = torch.cat((x_in, x_out), dim = 1)
            c = torch.sigmoid(torch.einsum('i, mi->m', self.D, x_all) + self.b)

            return torch.einsum('m, mi->mi', c, x_in) + torch.einsum('m, mi->mi',1 - c, x_out)
        else:
            return self.model(input)

class Symmetrize2D(nn.Module):
    '''
    https://github.com/calico/basenji/blob/master/basenji/layers.py

    Symmetrises input by computing (x + x.T) / 2'''
    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def forward(self, x):

        if len(x.shape) == 4:
            # assume x is of shape N x C x H x W
            return  (x + torch.transpose(x, 2, 3)) / 2
        elif len(x.shape) == 3:
            # assume x is of shape N x H x W
            return  (x + torch.transpose(x, 1, 2)) / 2
        elif len(x.shape) == 2:
            # assume x is a matrix of parameters
            return (x + x.t()) / 2
        else:
            raise Exception('Invalid shape: {}'.format(x.shape))

class Bilinear(nn.Module):
    def __init__(self):
        super(Bilinear, self).__init__()

    def forward(self, x, w):
        if len(W.shape) == 2:
            # W is fixed per batch
            left = torch.einsum('nij,jk->nik', x, W)
        elif len(W.shape) == 3:
            # W varies with batch
            left = torch.einsum('nij,njk->nik', x, W)
        return torch.einsum('nik,njk->nij', left, L_out)


class AverageTo2d(nn.Module):
    '''https://github.com/calico/basenji/blob/master/basenji/layers.py'''
    def __init__(self, concat_d = False, n = None, mode = 'avg'):
        """
        Inputs:
            concat_d: True if positional encoding should be appended
            n: spatial dimension
            mode: 'average' for default mode, 'concat' to concat instead of average, 'outer' to use outer product
        """
        super(AverageTo2d, self).__init__()
        assert mode in self.mode_options, 'Invalid mode: {}'.format(mode)
        self.concat_d = concat_d
        self.mode = mode
        if concat_d:
            self.get_positional_encoding(n)

    def get_positional_encoding(self, n):
        assert n is not None
        d = torch.zeros((n, n))
        for i in range(1, n):
            y = torch.diagonal(d, offset = i)
            y[:] = torch.ones(n-i) * i
        d = d + torch.transpose(d, 0, 1)
        d = torch.unsqueeze(d, 0)
        d = torch.unsqueeze(d, 0)
        self.d = d

    @property
    def mode_options(self):
        return {'avg', 'concat', 'outer', 'concat-outer', 'avg-outer', None}

    def forward(self, x):
        # assume x is of shape N x C x m
        # memory expensive
        assert len(x.shape) == 3, "shape must be 3D"
        N, C, m = x.shape

        out_list = []
        for mode in self.mode.split('-'):
            if mode is None:
                print("Warning: mode is None")
                # code will probably break if you get here
                out = x
            elif mode == 'avg':
                x1 = torch.tile(x, (1, 1, m))
                x1 = torch.reshape(x1, (-1, C, m, m))
                x2 = torch.transpose(x1, 2, 3)
                x1 = torch.unsqueeze(x1, 0)
                x2 = torch.unsqueeze(x2, 0)
                out = torch.cat((x1, x2), dim = 0)
                out = torch.mean(out, dim = 0, keepdim = False)
            elif mode == 'concat':
                x1 = torch.tile(x, (1, 1, m))
                x1 = torch.reshape(x1, (-1, C, m, m))
                x2 = torch.transpose(x1, 2, 3)
                out = torch.cat((x1, x2), dim = 1)
            elif mode == 'outer':
                # see test_average_to_2d_outer for evidence that this works
                x1 = torch.tile(x, (1, C, m))
                x1 = torch.reshape(x1, (-1, C*C, m, m))
                x2 = torch.transpose(x1, 2, 3)

                # use indices to permute x2
                indices = []
                for i in range(C):
                    indices.extend(range(i, i + C * (C-1) + 1, C))
                indices = torch.tensor(indices)
                if x2.is_cuda:
                    indices = indices.to(x2.get_device())
                x2 = torch.index_select(x2, dim = 1, index = indices)

                out = torch.einsum('ijkl,ijkl->ijkl', x1, x2)

            del x1, x2
            out_list.append(out)

        if self.concat_d:
            # append abs(i - j)
            if out.is_cuda:
                self.d = self.d.to(out.get_device())
            out_list.append(torch.tile(self.d, (N, 1, 1, 1)))

        out = torch.cat(out_list, dim = 1)
        return out

class FillDiagonalsFromArray(nn.Module):
    '''
    Uses input rank 1 tensor to fill all diagonals of output rank 2 tensor.

    Ouput[i,j] = input[i-j]
    '''
    def __init__(self):
        super(FillDiagonalsFromArray, self).__init__()

    def forward(self, input):
        if len(input.shape) == 2:
            # asume input is of shape N x m
            N, m = input.shape
        elif len(input.shape) == 1:
            # assume input is of shape m
            N = 1
            m, = input.shape
            input = input.reshape(N, m)

        if N == 1:
            # this is faster, but I didn't get it to work for N > 1
            # need to figure out the triu_indices for rank 3 tensors
            input = input.reshape(-1)
            output = torch.zeros((m, m), device = input.device)
            lengths = torch.arange(m, 0, -1)
            flattened = torch.cat([input[0:l] for l in lengths], dim = 0)
            output[np.triu_indices(m)] = flattened
            output = output.unsqueeze(0)
        else:
            output = torch.zeros((N, m, m), device = input.device)
            for n in range(N):
                for d in range(m):
                    rng = np.arange(m-d)
                    output[n, rng, rng+d] = input[n, d]

        output += torch.transpose(torch.triu(output, 1), 1, 2)
        return output

class FillDiagonalsFromArray2(nn.Module):
    '''
    Uses input rank 1 tensor to fill all diagonals of output rank 2 tensor.

    Ouput[i,j] = input[i-j]
    This version only works on cpu
    '''
    def __init__(self):
        super(FillDiagonalsFromArray2, self).__init__()

    def forward(self, input):
        if len(input.shape) == 2:
            # asume input is of shape N x m
            N, m = input.shape
        elif len(input.shape) == 1:
            # assume input is of shape m
            N = 1
            m, = input.shape
            input = input.reshape(N, m)

        output = np.zeros((N, m, m))
        for n in range(N):
            output[n] = scipy.linalg.toeplitz(input[n, :])

        output = torch.tensor(output, device = input.device, dtype = input.dtype)
        return output



def test_average_to_2d_outer():
    avg = AverageTo2d(mode = 'outer', concat_d = False, n = 10)
    verbose = True
    N = 1
    m = 10
    C_list = [3]
    for C in C_list:
        t0 = time.time()
        input = np.random.randint(low = 1, high = 10, size = (N, C, m))
        input = torch.tensor(input, dtype = torch.float32)
        if verbose:
            print(input[:, :, 0], input[:, :, 1])
        out1 = avg(input)
        if verbose:
            print(out1[:, :, 0, 1])
        tf = time.time()
        deltaT = np.round(tf - t0, 3)
        print("AverageTo2d time: {}".format(deltaT))

        t0 = time.time()
        out = np.zeros((N, C*C, m, m))
        for n in range(N):
            for i in range(m):
                for j in range(m):
                    out[n, :, i,j] = torch.flatten(torch.outer(input[n, :, i], input[n, :, j]))
        tf = time.time()
        deltaT = np.round(tf - t0, 3)
        print("For loop time: {}".format(deltaT))

        assert np.array_equal(out1, out), print(out1)

def test_MLP():
    mlp = MLP(5, [10, 10], 'batch', 'relu', 'prelu', dropout = True)
    print(mlp)
    x = torch.randn(2, 5)
    print(x, x.shape)
    out = mlp(x)
    print(out)

def test_FillDiagonalsFromArray():
    fill = FillDiagonalsFromArray()
    verbose = True
    N = 1
    m = 1024
    times = 100
    input = torch.tile(torch.range(0, m-1, 1), (N, 1)).to('cpu')
    t0 = time.time()
    for _ in range(times):
        out1 = fill(input)
        # if verbose:
        #     print(out1, out1.shape)

    tf = time.time()
    deltaT = np.round(tf - t0, 3)
    print("time: {}".format(deltaT))

    fill2 = FillDiagonalsFromArray2()
    t0 = time.time()
    for _ in range(times):
        out2 = fill2(input)
    tf = time.time()
    deltaT = np.round(tf - t0, 3)
    print("time: {}".format(deltaT))
    if verbose:
        print(out2.dtype, out2.shape, out2.device)

    assert torch.equal(out1, out2)


    # input = input.numpy().reshape(-1)
    # out = out1.numpy().reshape(N, m,m)
    # print(input, input.shape)
    # print(out)
    # print(out[:, np.triu_indices(m)])

    # input = np.arange(5)
    # input = input.reshape(-1)
    # output = np.zeros((m, m))
    # lengths = np.arange(m, 0, -1)
    # print(lengths)
    # flattened = np.concatenate([input[0:l] for l in lengths])
    # print('f', flattened)
    # output[np.triu_indices(m)] = flattened
    #
    # output +=np.triu(output, 1).T
    # print(output)

def test_strided():
    from numpy.lib.stride_tricks import as_strided
    d = np.array([1,2,3])
    m = len(d)

    vals = np.concatenate((d[::-1], d[1:]))
    print(vals)
    n = vals.strides[0]
    print(n)
    out = as_strided(vals[len(d)-1:], shape=(m,m), strides=(-n, n)).copy()
    print(out)

    print('-'*10)

    d = torch.tensor(d)
    inv_idx = torch.arange(d.size(0)-1, -1, -1).long()
    d_rev = d[inv_idx]
    vals = torch.cat((d_rev, d[1:]),)
    print(vals)
    n = vals.stride(0)
    print(n)
    out = torch.as_strided(vals[len(d)-1:], size=(m,m), stride=(-n, n))
    print(out)

def test_unpool():
    input = np.array([[1,2], [2,4]])
    input = torch.tensor(input, dtype=torch.float32).reshape(1, 1, 2, 2)
    print(input, input.shape)
    out = F.interpolate(input, size = (4, 4))
    print(out)

def test_triu_to_full():
    inp = torch.tensor(np.array([1,2,3,4,5,6]), dtype = torch.float32)
    out = torch_triu_to_full(inp)
    print(out)



if __name__ == '__main__':
    # test_average_to_2d_outer()
    # test_MLP()
    # test_FillDiagonalsFromArray()
    # test_strided()
    # test_unpool()
    test_triu_to_full()
