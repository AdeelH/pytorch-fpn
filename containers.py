from typing import Tuple, Callable, Any, Union

from torch import nn


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs: Union[Any, list, tuple]) -> tuple:
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
    Takes in an 2-tuple of the form
    (last_out, (1st input, 2nd input, ..., nth input))
    and passes it through the architecture shown below, returning a tuple
    of all outputs: (1st out, 2nd out, ..., nth out).

    In words: the ith layer in this sequential takes in as inputs the
    ith input and the output of the last layer i.e. the (i-1)th layer.
    For the 1st layer, the "output of the last layer" is last_out.

                      last_out
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

    def forward(self, inps: Tuple[Any, Tuple[Callable]]) -> tuple:
        last_out, layer_inps = inps
        outs = [None] * len(self)
        for i, (module, layer_inp) in enumerate(zip(self, layer_inps)):
            last_out = module((last_out, layer_inp))
            outs[i] = last_out
        return tuple(outs)
