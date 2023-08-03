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
        assert mode in self.mode_options, f'Invalid mode: {mode}'
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
        # (N = batches, C = channels, m = nodes)
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
