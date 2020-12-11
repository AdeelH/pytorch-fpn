from typing import Tuple, Sequence, Optional, Iterable

import torch
from torch import nn
import torchvision as tv

from containers import (Parallel, SequentialMultiInputMultiOutput,
                        SequentialMultiOutput)
from layers import (Residual, Interpolate, Reverse, AddTensors, SelectOne,
                    AddAcross, SplitTensor)


class FPN(nn.Sequential):
    """
    Implementation of the architecture described in
    "Feature Pyramid Networks for Object Detection" by Lin et al.,
    https://arxiv.com/abs/1612.03144.

    Takes in an n-tuple of feature maps in reverse order
    (nth feature map, (n-1)th feature map, ..., 1st feature map), where
    the nth feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    n outputs, such that the height and width of the ith output is equal to
    that of the corresponding input feature map and the number of channels
    is equal to out_channels.

    Returns all outputs as a tuple in the order:
    (nth out, (n-1)th out, ..., 1st out)

    1st feat. map ────[1st in_conv]──────┬─────[1st out_conv]────> 1st out
                                         │
                                     [upsample]
                                         │
                                         V
    2nd feat. map ────[2nd in_conv]────>(+)────[2nd out_conv]────> 2nd out
                                         │
                                     [upsample]
                                         │
                                         V
            .               .                        .                .
            .               .                        .                .
            .               .                        .                .
                                         │
                                     [upsample]
                                         │
                                         V
    nth feat. map ────[nth in_conv]────>(+)────[nth out_conv]────> nth out

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
        upsample_and_add = SequentialMultiInputMultiOutput(*[
            Residual(
                Interpolate(size=s[2:], mode='bilinear', align_corners=False))
            for s in in_feats_shapes
        ])
        out_convs = Parallel([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
            for s in in_feats_shapes
        ])
        # yapf: disable
        layers = [
            Reverse(),
            in_convs,
            upsample_and_add,
            out_convs,
            Reverse()
        ]
        # yapf: enable
        super().__init__(*layers)


class PanopticFPN(nn.Sequential):
    """
    Implementation of the architecture described in
    "Panoptic Feature Pyramid Networks" by Kirilov et al.,
    https://arxiv.com/abs/1901.02446.

    Takes in an n-tuple of feature maps in reverse order
    (nth feature map, (n-1)th feature map, ..., 1st feature map), where
    the nth feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    a single final output, with out_channels channels.

    1st feat. map ──[1st in_conv]──>──[1st upsampler]──────┐
                                                           │
                                                           │
                                                           V
    2nd feat. map ──[2nd in_conv]──>──[2nd upsampler]────>(+)
                                                           │
                                                           │
                                                           V
          .               .                  .             .
          .               .                  .             .
          .               .                  .             .
                                                           │
                                                           │
                                                           V
    nth feat. map ──[nth in_conv]──>──[nth upsampler]────>(+)
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
            c=hidden_channels,
            size=out_size,
            num_upsamples_per_layer=num_upsamples_per_layer,
            num_groups=num_groups_for_norm)
        out_conv = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)

        # yapf: disable
        layers = [
            in_convs,
            upsamplers,
            AddTensors(),
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
            upsample_layer = Interpolate(
                scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            upsample_layer = Interpolate(
                size=size, mode='bilinear', align_corners=False)

        conv_block.append(upsample_layer)
        return nn.Sequential(*conv_block)


class PANetFPN(nn.Sequential):
    def __init__(self,
                 in_feats_shapes: list,
                 hidden_channels: int = 256,
                 out_channels: int = 2):
        fpn1 = FPN(
            in_feats_shapes,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels)
        in_feats_shapes = [(n, hidden_channels, h, w)
                           for (n, c, h, w) in in_feats_shapes]
        fpn2 = FPN(
            in_feats_shapes[::-1],
            hidden_channels=hidden_channels,
            out_channels=out_channels)
        # yapf: disable
        layers = [
            fpn1,
            Reverse(),
            fpn2,
            Reverse(),
        ]
        # yapf: enable
        super().__init__(*layers)


def _get_shapes(m, ch=3, sz=224):
    state = m.training
    m.eval()
    with torch.no_grad():
        feats = m(torch.empty(1, ch, sz, sz))
    m.train(state)
    return [f.shape for f in feats]


class EfficientNetFeatureMapsExtractor(nn.Module):
    def __init__(self, effnet):
        super().__init__()
        self.m = effnet

    def forward(self, x):
        feats = self.m.extract_endpoints(x)
        return list(feats.values())


class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model, mode=None):
        super().__init__()
        self.mode = mode
        # yapf: disable
        stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        layers = [
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        # yapf: enable
        if mode == 'fusion':
            self.m = nn.Sequential(
                Parallel([stem, nn.Identity()]),
                SequentialMultiInputMultiOutput(
                    *[nn.Sequential(AddTensors(), m) for m in layers]))
        else:
            self.m = SequentialMultiOutput(stem, *layers)

    def forward(self, x):
        if self.mode != 'fusion':
            return self.m(x)
        x, inps = x
        return self.m((x, inps))


def _load_efficientnet(name,
                       num_classes=1000,
                       pretrained='imagenet',
                       in_channels=3):
    model = torch.hub.load(
        'lukemelas/EfficientNet-PyTorch',
        name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels)
    return model


def make_segm_fpn_efficientnet(name='efficientnet_b0',
                               fpn_type='fpn',
                               out_size=(224, 224),
                               fpn_channels=256,
                               num_classes=1000,
                               pretrained='imagenet',
                               in_channels=3):
    effnet = _load_efficientnet(
        name=name, num_classes=num_classes, pretrained=pretrained)
    if in_channels > 3:
        new_channels = in_channels - 3
        new_effnet = _load_efficientnet(
            name=name,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=new_channels,
        )
        backbone = nn.Sequential(
            SplitTensor(size_or_sizes=(3, new_channels), dim=1),
            Parallel([
                EfficientNetFeatureMapsExtractor(effnet),
                EfficientNetFeatureMapsExtractor(new_effnet)
            ]), AddAcross())
    else:
        backbone = EfficientNetFeatureMapsExtractor(effnet)

    feats_shapes = _get_shapes(backbone, ch=in_channels, sz=out_size[0])
    if fpn_type == 'fpn':
        fpn = nn.Sequential(
            FPN(feats_shapes,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    elif fpn_type == 'panoptic':
        fpn = PanopticFPN(
            feats_shapes,
            hidden_channels=fpn_channels,
            out_channels=num_classes)
    elif fpn_type == 'panet+fpn':
        feats_shapes2 = [(n, fpn_channels, h, w)
                         for (n, c, h, w) in feats_shapes]
        fpn = nn.Sequential(
            PANetFPN(
                feats_shapes,
                hidden_channels=fpn_channels,
                out_channels=fpn_channels),
            FPN(feats_shapes2,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    else:
        raise NotImplementedError()

    model = nn.Sequential(
        backbone, fpn,
        Interpolate(size=out_size, mode='bilinear', align_corners=True))
    return model


def make_fusion_resnet_backbone(old_resnet: nn.Module,
                                new_resnet: nn.Module) -> nn.Module:
    """ Create a parallel backbone with multi-point fusion. """
    new_conv = new_resnet.conv1

    backbone = nn.Sequential(
        SplitTensor(size_or_sizes=(3, new_conv.in_channels), dim=1),
        Parallel([nn.Identity(),
                  ResNetFeatureMapsExtractor(new_resnet)]),
        ResNetFeatureMapsExtractor(old_resnet, mode='fusion'))
    return backbone


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


def make_segm_fpn_resnet(name: str = 'resnet18',
                         fpn_type: str = 'fpn',
                         out_size: Tuple[int] = (224, 224),
                         fpn_channels: int = 256,
                         num_classes: int = 1000,
                         pretrained: bool = True,
                         in_channels: int = 3) -> nn.Module:
    assert in_channels > 0
    assert num_classes > 0
    assert out_size[0] > 0 and out_size[1] > 0

    resnet = tv.models.resnet.__dict__[name](pretrained=pretrained)
    if in_channels == 3:
        backbone = ResNetFeatureMapsExtractor(resnet)
    else:
        old_conv = resnet.conv1
        old_conv_args = {
            'out_channels': old_conv.out_channels,
            'kernel_size': old_conv.kernel_size,
            'stride': old_conv.stride,
            'padding': old_conv.padding,
            'dilation': old_conv.dilation,
            'groups': old_conv.groups,
            'bias': old_conv.bias
        }
        if not pretrained:
            # just replace the first conv layer
            new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
            resnet.conv1 = new_conv
            backbone = ResNetFeatureMapsExtractor(resnet)
        else:
            if in_channels > 3:
                new_channels = in_channels - 3
                new_conv = nn.Conv2d(in_channels=new_channels, **old_conv_args)

                resnet_constructor = tv.models.resnet.__dict__[name]
                new_resnet = resnet_constructor(pretrained=pretrained)
                new_resnet.conv1 = copy_conv_weights(old_conv, new_conv)

                backbone = make_fusion_resnet_backbone(resnet, new_resnet)
            else:
                new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
                resnet.conv1 = copy_conv_weights(old_conv, new_conv)
                backbone = ResNetFeatureMapsExtractor(resnet)

    feats_shapes = _get_shapes(backbone, ch=in_channels, sz=out_size[0])
    if fpn_type == 'fpn':
        fpn = nn.Sequential(
            FPN(feats_shapes,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    elif fpn_type == 'panoptic':
        fpn = PanopticFPN(
            feats_shapes,
            hidden_channels=fpn_channels,
            out_channels=num_classes)
    elif fpn_type == 'panet+fpn':
        feats_shapes2 = [(n, fpn_channels, h, w)
                         for (n, c, h, w) in feats_shapes]
        fpn = nn.Sequential(
            PANetFPN(
                feats_shapes,
                hidden_channels=fpn_channels,
                out_channels=fpn_channels),
            FPN(feats_shapes2,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    else:
        raise NotImplementedError()

    # yapf: disable
    model = nn.Sequential(
        backbone,
        fpn,
        Interpolate(size=out_size, mode='bilinear', align_corners=True))
    # yapf: enable
    return model
