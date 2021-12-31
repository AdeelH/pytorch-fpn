from typing import Tuple, Sequence, Optional, Iterable

from torch import nn

from fpn.containers import (Parallel, SequentialMultiInputMultiOutput)
from fpn.layers import (Interpolate, Reverse, Sum)


class FPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Feature Pyramid Networks for Object Detection" by Lin et al.,
    https://arxiv.com/abs/1612.03144.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    n outputs, such that the height and width of the ith output is equal to
    that of the corresponding input feature map and the number of channels
    is equal to out_channels.

    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)

    Architecture diagram:

    nth feat. map ────────[nth in_conv]──────────┐────────[nth out_conv]────> nth out
                                                 │
                                             [upsample]
                                                 │
                                                 V
    (n-1)th feat. map ────[(n-1)th in_conv]────>(+)────[(n-1)th out_conv]────> (n-1)th out
                                                 │
                                             [upsample]
                                                 │
                                                 V
            .                     .                           .                    .
            .                     .                           .                    .
            .                     .                           .                    .
                                                 │
                                             [upsample]
                                                 │
                                                 V
    1st feat. map ────────[1st in_conv]────────>(+)────────[1st out_conv]────> 1st out

    """

    def __init__(self,
                 in_feats_shapes: Sequence[Tuple[int, ...]],
                 hidden_channels: int = 256,
                 out_channels: int = 2):
        """Constructor.

        Args:
            in_feats_shapes (Sequence[Tuple[int, ...]]): Shapes of the feature
                maps that will be fed into the network. These are expected to
                be tuples of the form (., C, H, ...).
            hidden_channels (int, optional): The number of channels to which
                all feature maps are convereted before being added together.
                Defaults to 256.
            out_channels (int, optional): Number of output channels. This will
                normally be the number of classes. Defaults to 2.
        """
        # reverse so that the deepest (i.e. produced by the deepest layer in
        # the backbone network) feature map is first.
        in_feats_shapes = in_feats_shapes[::-1]
        in_feats_channels = [s[1] for s in in_feats_shapes]

        # 1x1 conv to make the channels of all feature maps the same
        in_convs = Parallel([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            for in_channels in in_feats_channels
        ])

        # yapf: disable
        def resize_and_add(to_size):
            return nn.Sequential(
                Parallel([nn.Identity(), Interpolate(size=to_size)]),
                Sum()
            )

        top_down_layer = SequentialMultiInputMultiOutput(
            nn.Identity(),
            *[resize_and_add(shape[-2:]) for shape in in_feats_shapes[1:]]
        )

        out_convs = Parallel([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_feats_shapes
        ])
        layers = [
            Reverse(),
            in_convs,
            top_down_layer,
            out_convs,
            Reverse()
        ]
        # yapf: enable
        super().__init__(*layers)


class PanopticFPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Panoptic Feature Pyramid Networks" by Kirilov et al.,
    https://arxiv.com/abs/1901.02446.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    a single final output, with out_channels channels.

    Architecture diagram:

    nth feat. map ────[nth in_conv]─────────>───[nth upsampler]──────────┐
                                                                         │
                                                                         │
                                                                         V
    (n-1)th feat. map ──[(n-1)th in_conv]───>───[(n-1)th upsampler]────>(+)
                                                                         │
                                                                         │
                                                                         V
          .                     .                     .
          .                     .                     .
          .                     .                     .
                                                                         │
                                                                         │
                                                                         V
    1st feat. map ────[1st in_conv]─────────>───[1st upsampler]─────────(+)
                                                                         │
                                                                         │
                                                                         V
                                                                        out
    """

    def __init__(self,
                 in_feats_shapes: Sequence[Tuple[int, ...]],
                 hidden_channels: int = 256,
                 out_channels: int = 2,
                 out_size: Optional[int] = None,
                 num_upsamples_per_layer: Optional[Sequence[int]] = None,
                 upsamplng_factor: int = 2,
                 num_groups_for_norm: int = 32):
        """Constructor.

        Args:
            in_feats_shapes (Sequence[Tuple[int, ...]]): Shapes of the feature
                maps that will be fed into the network. These are expected to
                be tuples of the form (., C, H, ...).
            hidden_channels (int, optional): The number of channels to which
                all feature maps are convereted before being added together.
                Defaults to 256.
            out_channels (int, optional): Number of output channels. This will
                normally be the number of classes. Defaults to 2.
            out_size (Optional[int], optional): Size of output. If None, 
                the size of the first feature map will be used.
                Defaults to None.
            num_upsamples_per_layer (Optional[Sequence[int]], optional): Number
                of upsampling iterations for each feature map. Will depend on
                the size of the feature map. Each upsampling iteration
                comprises a conv-group_norm-relu block followed by a scaling
                using torch.nn.functional.interpolate.
                If None, each feature map is assumed to be half the size of the
                preceeding one, meaning that it requires one more upsampling
                iteration than the last one.
                Defaults to None.
            upsamplng_factor (int, optional): How much to scale per upsampling
                iteration. Defaults to 2.
            num_groups_for_norm (int, optional): Number of groups for group
                norm layers. Defaults to 32.
        """
        if num_upsamples_per_layer is None:
            num_upsamples_per_layer = list(range(len(in_feats_shapes)))

        if out_size is None:
            out_size = in_feats_shapes[0][-2:]

        in_convs = Parallel([
            nn.Conv2d(s[1], hidden_channels, kernel_size=1)
            for s in in_feats_shapes
        ])
        upsamplers = self._make_upsamplers(
            in_channels=hidden_channels,
            size=out_size,
            num_upsamples_per_layer=num_upsamples_per_layer,
            num_groups=num_groups_for_norm)
        out_conv = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)

        # yapf: disable
        layers = [
            in_convs,
            upsamplers,
            Sum(),
            out_conv
        ]
        # yapf: enable
        super().__init__(*layers)

    @classmethod
    def _make_upsamplers(cls,
                         in_channels: int,
                         size: int,
                         num_upsamples_per_layer: Iterable[int],
                         num_groups: int = 32) -> Parallel:
        layers = []
        for num_upsamples in num_upsamples_per_layer:
            upsampler = cls._upsample_feat(
                in_channels=in_channels,
                num_upsamples=num_upsamples,
                size=size,
                num_groups=num_groups)
            layers.append(upsampler)

        upsamplers = Parallel(layers)
        return upsamplers

    @classmethod
    def _upsample_feat(cls,
                       in_channels: int,
                       num_upsamples: int,
                       size: int,
                       scale_factor: float = 2.,
                       num_groups: int = 32) -> nn.Sequential:
        if num_upsamples == 0:
            return cls._make_upsampling_block(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                scale=1,
                num_groups=num_groups)
        blocks = []
        for _ in range(num_upsamples - 1):
            blocks.append(
                cls._make_upsampling_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    scale=scale_factor,
                    num_groups=num_groups))
        blocks.append(
            cls._make_upsampling_block(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                size=size,
                num_groups=num_groups))
        return nn.Sequential(*blocks)

    @classmethod
    def _make_upsampling_block(cls,
                               in_channels: int,
                               out_channels: int = None,
                               scale: float = 2,
                               size: int = None,
                               num_groups: int = 32) -> nn.Sequential:
        if out_channels is None:
            out_channels = in_channels

        # conv block that preserves size
        conv_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels=out_channels, num_groups=num_groups),
            nn.ReLU(inplace=True)
        ]
        if scale == 1:
            # don't upsample
            return nn.Sequential(*conv_block)

        if size is None:
            upsample_layer = Interpolate(scale_factor=scale)
        else:
            upsample_layer = Interpolate(size=size)

        conv_block.append(upsample_layer)
        return nn.Sequential(*conv_block)


class PANetFPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Path Aggregation Network for Instance Segmentation" by Liu et al.,
    https://arxiv.com/abs/1803.01534. This architecture adds a bottom-up path
    after the top-down path in a normal FPN. It can be thought of as a normal
    FPN followed by a flipped FPN.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    n outputs, such that the height and width of the ith output is equal to
    that of the corresponding input feature map and the number of channels
    is equal to out_channels.

    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)

    Architecture diagram:

            (1st feature map, 2nd feature map, ..., nth feature map)
                                    │
                                [1st FPN]
                                    │
                                    V
                                    │
                        [Reverse the order of outputs]
                                    │
                                    V
                                    │
                                [2nd FPN]
                                    │
                                    V
                                    │
                        [Reverse the order of outputs]
                                    │
                                    │
                                    V
                       (1st out, 2nd out, ..., nth out)

    """

    def __init__(self, fpn1: nn.Module, fpn2: nn.Module):
        # yapf: disable
        layers = [
            fpn1,
            Reverse(),
            fpn2,
            Reverse(),
        ]
        # yapf: enable
        super().__init__(*layers)
