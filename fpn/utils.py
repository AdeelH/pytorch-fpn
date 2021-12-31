from typing import Tuple, List

import torch
from torch import nn


def copy_conv_weights(src_conv: nn.Conv2d,
                      dst_conv: nn.Conv2d,
                      dst_start_idx: int = 0) -> nn.Module:
    """Copy weights from one convolution layer to another. Repeat if the
    destination layer has more channels than the source layer.

    Args:
        src_conv (nn.Conv2d): Conv layer from which weights will be copied.
        dst_conv (nn.Conv2d): Conv layer to which weights will be copied.
        dst_start_idx (int, optional): Index at which to start copying.
            Defaults to 0.

    Returns:
        nn.Module: dst_conv with updated weights
    """
    src_channels = src_conv.in_channels
    dst_channels = dst_conv.in_channels

    for dst_idx in range(dst_start_idx, dst_channels):
        src_idx = dst_idx % src_channels - dst_start_idx
        weights = src_conv.weight.data[:, src_idx]
        dst_conv.weight.data[:, dst_idx] = weights

    return dst_conv


def _get_shapes(model: nn.Module,
                channels: int = 3,
                size: Tuple[int, int] = (224, 224)) -> List[Tuple[int, ...]]:
    """Extract shapes of feature maps computed by the model.

    The model must be an nn.Module whose __call__ method returns all feature
    maps when called with an input.
    """
    # save state so we can restore laterD
    state = model.training

    model.eval()
    with torch.no_grad():
        x = torch.empty(1, channels, *size)
        feats = model(x)

    # restore state
    model.train(state)

    if isinstance(feats, torch.Tensor):
        feats = [feats]

    feat_shapes = [f.shape for f in feats]
    return feat_shapes
