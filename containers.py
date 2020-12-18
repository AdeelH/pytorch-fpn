from typing import Any

from torch import nn


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs: Any) -> tuple:
        # if multiple inputs, pass the 1st input through the 1st module,
        # the 2nd input through the 2nd module, and so on.
        if isinstance(xs, (list, tuple)):
            return tuple(m(x) for m, x in zip(self, xs))
        # if single input, pass it through all modules
        return tuple(m(xs) for m in self)


class SequentialMultiOutput(nn.Sequential):
    """
    Like nn.Squential but returns all intermediate outputs as a tuple.

      input
        │
        │
        V
    [1st layer]───────> 1st out
        │
        │
        V
    [2nd layer]───────> 2nd out
        │
        │
        V
        .
        .
        .
        │
        │
        V
    [nth layer]───────> nth out

    """

    def forward(self, x: Any) -> tuple:
        outs = [None] * len(self)
        last_out = x
        for i, module in enumerate(self):
            last_out = module(last_out)
            outs[i] = last_out
        return tuple(outs)


class SequentialMultiInputMultiOutput(nn.Sequential):
    """
    Takes in either
    (1) an (n+1)-tuple of the form
      (last_out, 1st input, 2nd input, ..., nth input), or
    (2) an n-tuple of the form
      (1st input, 2nd input, ..., nth input),
    where n is the length of this sequential.

    If (2), the first layer in this sequential should be able to accept
    a single input. All others are expected to accept a 2-tuple of inputs.

    Returns an n-tuple of all outputs of the form:
    (1st out, 2nd out, ..., nth out).

    In other words: the ith layer in this sequential takes in as inputs the
    ith input and the output of the last layer i.e. the (i-1)th layer.
    For the 1st layer, the "output of the last layer" is last_out.

                       last_out
                      (optional)
                          │
                          │
                          V
    1st input ───────[1st layer]───────> 1st out
                          │
                          │
                          V
    2nd input ───────[2nd layer]───────> 2nd out
                          │
                          │
                          V
        .                 .                  .
        .                 .                  .
        .                 .                  .
                          │
                          │
                          V
    nth input ───────[nth layer]───────> nth out

    """

    def forward(self, xs: tuple) -> tuple:
        outs = [None] * len(self)

        if len(xs) == len(self) + 1:
            last_out = xs[0]
            layer_inputs = xs[1:]
            layers = self
            start_idx = 0
        elif len(xs) == len(self):
            last_out = self[0](xs[0])
            layer_inputs = xs[1:]
            layers = self[1:]
            outs[0] = last_out
            start_idx = 1
        else:
            raise ValueError('Invalid input format.')

        for i, (layer, x) in enumerate(zip(layers, layer_inputs), start_idx):
            last_out = layer((x, last_out))
            outs[i] = last_out

        return tuple(outs)
