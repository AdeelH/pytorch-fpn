from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from containers import Parallel


class Residual(nn.Sequential):
    """Pass the input through a layer and add the result with the input."""

    def __init__(self, layer):
        # yapf: disable
        layers = [
            Parallel([nn.Identity(), layer]),
            Sum()
        ]
        # yapf: enable
        super().__init__(*layers)


class ModulizedFunction(nn.Module):
    """Convert a function to an nn.Module."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = partial(fn, *args, **kwargs)

    def forward(self, x):
        return self.fn(x)


class Interpolate(ModulizedFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(F.interpolate, *args, **kwargs)


class SplitTensor(ModulizedFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.split, *args, **kwargs)


class Sum(nn.Module):
    def forward(self, inps):
        return sum(inps)


class AddAcross(nn.Module):
    def forward(self, inps):
        return [sum(items) for items in zip(*inps)]


class Reverse(nn.Module):
    def forward(self, inps):
        return inps[::-1]


class SelectOne(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        return xs[self.idx]


class Debug(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}')
        print(f'type: {type(x)}, len: {len(x)}')
        print(f'shapes: {self._get_shape_recurse(x)}')
        return x

    def _get_shape_recurse(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape
        return [self._get_shape_recurse(a) for a in x]
