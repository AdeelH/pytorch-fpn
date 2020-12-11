from typing import Tuple, List, Iterable

import torch
from torch import nn


def copy_conv_weights(src_conv: nn.Conv2d,
                      dst_conv: nn.Conv2d,
                      dst_start_idx: int = 0) -> nn.Module:
    src_channels = src_conv.in_channels
    dst_channels = dst_conv.in_channels - dst_start_idx

    remaining_channels = dst_channels
    i = dst_start_idx
    while remaining_channels > 0:
        chunk_size = min(remaining_channels, src_channels)
        pt_weights = src_conv.weight.data[:, :chunk_size]
        dst_conv.weight.data[:, i:i + chunk_size] = pt_weights
        i += chunk_size
        remaining_channels -= chunk_size
    return dst_conv


def _get_shapes(model: nn.Module,
                channels: int = 3,
                size: Tuple[int, int] = (224, 224)) -> List[Tuple[int, ...]]:
    """Extract shapes of feature maps returned by the model. The model must
    return all feature maps when called with an input.
    """
    state = model.training
    model.eval()
    with torch.no_grad():
        x = torch.empty(1, channels, *size)
        feats: Iterable[torch.Tensor] = model(x)
    model.train(state)
    return [f.shape for f in feats]
