from typing import Tuple, Optional

import torch
from torch import nn
import torchvision as tv

from fpn.containers import Parallel
from fpn.layers import (Interpolate, AddAcross, SplitTensor, SelectOne)
from fpn.fpn import (FPN, PanopticFPN, PANetFPN)
from fpn.utils import (copy_conv_weights, _get_shapes)
from fpn.backbone import (ResNetFeatureMapsExtractor, make_fusion_resnet_backbone,
                      EfficientNetFeatureMapsExtractor)


def make_fpn_resnet(name: str = 'resnet18',
                    fpn_type: str = 'fpn',
                    out_size: Tuple[int, int] = (224, 224),
                    fpn_channels: int = 256,
                    num_classes: int = 1000,
                    pretrained: bool = True,
                    in_channels: int = 3) -> nn.Module:
    """Create an FPN model with a ResNet backbone.

    If `in_channels > 3`, uses the fusion technique described in the paper,
    *FuseNet*, by Hazirbas et al.
    (https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf)
    that adds a parallel resnet backbone for the new channels. All the
    pretrained weights are retained.

    Args:
        name (str, optional): Name of the resnet backbone. Only those available
            in torchvision are supported. Defaults to 'resnet18'.
        fpn_type (str, optional): Type of FPN. 'fpn' | 'panoptic' | 'panet'.
            Defaults to 'fpn'.
        out_size (Tuple[int, int], optional): Size of segmentation output.
            Defaults to (224, 224).
        fpn_channels (int, optional): Number of hidden channels to use in the
            FPN. Defaults to 256.
        num_classes (int, optional): Number of classes for which to make
            predictions. Determines the channel width of the output.
            Defaults to 1000.
        pretrained (bool, optional): Whether to use pretrained backbone.
            Defaults to True.
        in_channels (int, optional): Channel width of the input. If less than
            3, conv1 is replaced with a smaller one. If greater than 3, a
            FuseNet-style architecture is used to incorporate the new channels.
            In both cases, pretrained weights are retained. Defaults to 3.

    Raises:
        NotImplementedError: On unknown fpn_style.

    Returns:
        nn.Module: the FPN model
    """
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

                backbone = make_fusion_resnet_backbone(resnet, new_resnet)
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
    elif fpn_type == 'panet':
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
        Interpolate(size=out_size))
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


def make_fpn_efficientnet(name: str = 'efficientnet_b0',
                          fpn_type: str = 'fpn',
                          out_size: Tuple[int, int] = (224, 224),
                          fpn_channels: int = 256,
                          num_classes: int = 1000,
                          pretrained: Optional[str] = 'imagenet',
                          in_channels: str = 3) -> nn.Module:
    """Loads the PyTorch implementation of EfficientNet from
    https://github.com/lukemelas/EfficientNet-PyTorch using torch.hub.

    Args:
        name (str, optional): Name of the EfficientNet backbone. Only those
            available in the lukemelas/EfficientNet-PyTorch repos are
            supported. Defaults to 'efficientnet_b0'.
        fpn_type (str, optional): Type of FPN. 'fpn' | 'panoptic' | 'panet'.
            Defaults to 'fpn'.
        out_size (Tuple[int, int], optional): Size of segmentation output.
            Defaults to (224, 224).
        fpn_channels (int, optional): Number of hidden channels to use in the
            FPN. Defaults to 256.
        num_classes (int, optional): Number of classes for which to make
            predictions. Determines the channel width of the output.
            Defaults to 1000.
        pretrained (Optional[str], optional): One of
            False | True | 'imagenet' | 'advprop'.
            See lukemelas/EfficientNet-PyTorch for details.
            Defaults to 'imagenet'.
        in_channels (int, optional): Channel width of the input. If greater
            than 3, a parallel backbone is added to incorporate the new
            channels and the feature maps of the two backbones are added
            together to produce the final feature maps. Note that this is
            currently different from make_fpn_resnet. See
            lukemelas/EfficientNet-PyTorch for the in_channels < 3 case.
            Defaults to 3.

    Raises:
        NotImplementedError: On unknown fpn_style.

    Returns:
        nn.Module: the FPN model
    """
    if in_channels <= 3:
        effnet = _load_efficientnet(
            name=name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained)
        backbone = EfficientNetFeatureMapsExtractor(effnet)
    else:
        effnet = _load_efficientnet(
            name=name,
            num_classes=num_classes,
            in_channels=3,
            pretrained=pretrained)
        new_channels = in_channels - 3
        new_effnet = _load_efficientnet(
            name=name,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=new_channels)
        backbone = nn.Sequential(
            SplitTensor((3, new_channels), dim=1),
            Parallel([
                EfficientNetFeatureMapsExtractor(effnet),
                EfficientNetFeatureMapsExtractor(new_effnet)
            ]), AddAcross())

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
    elif fpn_type == 'panet':
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
        Interpolate(size=out_size))
    # yapf: enable
    return model
