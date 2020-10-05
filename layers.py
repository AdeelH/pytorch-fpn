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


class AddTensors(nn.Module):
    def forward(self, inps):
        return sum(inps)


class Reverse(nn.Module):
    def forward(self, inps):
        return inps[::-1]


class SelectOne(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        return xs[self.idx]
