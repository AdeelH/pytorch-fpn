import torch
from torch import nn


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs):
        if isinstance(xs, torch.Tensor):
            return tuple(m(xs) for m in self)
        return tuple(m(x) for m, x in zip(self, xs))


class SequentialMultiOutput(nn.Sequential):
    def forward(self, x):
        outputs = [None] * len(self)
        out = x
        for i, module in enumerate(self):
            out = module(out)
            outputs[i] = out
        return outputs


class SequentialMultiInputMultiOutput(nn.Sequential):
    def forward(self, inps):
        outputs = [None] * len(self)
        out = self[0](inps[0])
        outputs[0] = out
        for i, (module, inp) in enumerate(zip(self[1:], inps[1:]), start=1):
            out = module((inp, out))
            outputs[i] = out
        return outputs
