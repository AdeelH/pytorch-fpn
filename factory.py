from typing import Tuple, Optional

import torch
from torch import nn
import torchvision as tv

from containers import Parallel
from layers import (Interpolate, AddAcross, SplitTensor, SelectOne)
from fpn import (FPN, PanopticFPN, PANetFPN)
from utils import (copy_conv_weights, _get_shapes)
from backbone import (ResNetFeatureMapsExtractor, make_fused_backbone,
                      EfficientNetFeatureMapsExtractor)


def make_segm_fpn_resnet(name: str = 'resnet18',
                         fpn_type: str = 'fpn',
                         out_size: Tuple[int, int] = (224, 224),
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

                resnet_cls = tv.models.resnet.__dict__[name]
                new_resnet = resnet_cls(pretrained=pretrained)
                new_resnet.conv1 = copy_conv_weights(old_conv, new_conv)

                backbone = make_fused_backbone(resnet, new_resnet)
            else:
                new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
                resnet.conv1 = copy_conv_weights(old_conv, new_conv)
                backbone = ResNetFeatureMapsExtractor(resnet)

    feat_shapes = _get_shapes(backbone, channels=in_channels, size=out_size)
    if fpn_type == 'fpn':
        fpn = nn.Sequential(
            FPN(feat_shapes,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    elif fpn_type == 'panoptic':
        fpn = PanopticFPN(
            feat_shapes,
            hidden_channels=fpn_channels,
            out_channels=num_classes)
    elif fpn_type == 'panet_fpn':
        fpn1 = FPN(
            feat_shapes,
            hidden_channels=fpn_channels,
            out_channels=fpn_channels)

        feat_shapes = [(n, fpn_channels, h, w) for (n, c, h, w) in feat_shapes]
        fpn2 = FPN(
            feat_shapes[::-1],
            hidden_channels=fpn_channels,
            out_channels=num_classes)
        fpn = nn.Sequential(PANetFPN(fpn1, fpn2), SelectOne(idx=0))
    else:
        raise NotImplementedError()

    # yapf: disable
    model = nn.Sequential(
        backbone,
        fpn,
        Interpolate(size=out_size, mode='bilinear', align_corners=True))
    # yapf: enable
    return model


def _load_efficientnet(name: str,
                       num_classes: int = 1000,
                       pretrained: Optional[str] = 'imagenet',
                       in_channels: int = 3) -> nn.Module:
    model = torch.hub.load(
        'lukemelas/EfficientNet-PyTorch',
        name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels)
    return model


def make_segm_fpn_efficientnet(name: str = 'efficientnet_b0',
                               fpn_type: str = 'fpn',
                               out_size: Tuple[int, int] = (224, 224),
                               fpn_channels: int = 256,
                               num_classes: int = 1000,
                               pretrained: Optional[str] = 'imagenet',
                               in_channels: str = 3) -> nn.Module:
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

    feat_shapes = _get_shapes(backbone, channels=in_channels, size=out_size)
    if fpn_type == 'fpn':
        fpn = nn.Sequential(
            FPN(feat_shapes,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    elif fpn_type == 'panoptic':
        fpn = PanopticFPN(
            feat_shapes,
            hidden_channels=fpn_channels,
            out_channels=num_classes)
    elif fpn_type == 'panet+fpn':
        feat_shapes2 = [(n, fpn_channels, h, w)
                        for (n, c, h, w) in feat_shapes]
        fpn = nn.Sequential(
            PANetFPN(
                feat_shapes,
                hidden_channels=fpn_channels,
                out_channels=fpn_channels),
            FPN(feat_shapes2,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    else:
        raise NotImplementedError()

    model = nn.Sequential(
        backbone, fpn,
        Interpolate(size=out_size, mode='bilinear', align_corners=False))
    return model
