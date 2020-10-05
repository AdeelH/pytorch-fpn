from functools import partial

from torch import nn
from torch.nn import functional as F

from containers import Parallel


class Residual(nn.Sequential):
    def __init__(self, layer):
        super().__init__(
            Parallel([nn.Identity(), layer]),
            AddTensors()
        )


class Interpolate(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fn = partial(F.interpolate, **kwargs)

    def forward(self, x):
        return self.fn(x)


class SplitTensor(nn.Module):
    """ Wrapper around `torch.split` """

    def __init__(self, size_or_sizes, dim):
        super().__init__()
        self.size_or_sizes = size_or_sizes
        self.dim = dim

    def forward(self, X):
        return X.split(self.size_or_sizes, dim=self.dim)


class AddTensors(nn.Module):
    def forward(self, inps):
        return sum(inps)


class AddAcross(nn.Module):
    def forward(self, inps):
        inps1, inps2 = inps
        return [i1 + i2 for i1, i2 in zip(inps1, inps2)]


class Reverse(nn.Module):
    def forward(self, inps):
        return inps[::-1]


class SelectOne(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        return xs[self.idx]
