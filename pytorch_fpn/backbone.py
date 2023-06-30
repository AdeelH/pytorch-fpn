from typing import Optional, Tuple, Type, Union

from torch import nn
import torchvision as tv

from pytorch_fpn.containers import (Parallel, SequentialMultiInputMultiOutput,
                                    SequentialMultiOutput)
from pytorch_fpn.layers import (Sum, SplitTensor)
from pytorch_fpn.utils import copy_conv_weights


class EfficientNetFeatureMapsExtractor(nn.Module):
    def __init__(self, effnet: nn.Module, mode: Optional[str] = None):
        super().__init__()
        self.m = effnet

        if mode is not None:
            # TODO implement
            raise NotImplementedError()

    def forward(self, x) -> tuple:
        feats = self.m.extract_endpoints(x)
        return tuple(feats.values())


class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model: nn.Module, mode: Optional[str] = None):
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
        if mode == 'fusion':
            # allow each layer to take in multiple inputs - summming them
            # before passing them through. This allows each layer to take in
            # in feature maps from multiple backbones.
            multi_input_layers = [
                nn.Sequential(Sum(), layer) for layer in layers
            ]
            self.m = SequentialMultiInputMultiOutput(
                stem,
                *multi_input_layers,
                Sum()
            )
        else:
            self.m = SequentialMultiOutput(stem, *layers)
        # yapf: enable

    def forward(self, x):
        if self.mode != 'fusion':
            return self.m(x)
        x, in_feats = x
        return self.m((x, *in_feats))


def make_fused_backbone(old_backbone: nn.Module, new_backbone: nn.Module,
                        featureMapExtractorCls: Type,
                        channel_split: Tuple[int, int]) -> nn.Module:
    """Create a fused backbone using FuseNet style feature fusion.
    See the paper, "FuseNet", by Hazirbas et al., 
    https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf.
    """
    backbone = nn.Sequential(
        SplitTensor(channel_split, dim=1),
        Parallel([nn.Identity(),
                  featureMapExtractorCls(new_backbone)]),
        featureMapExtractorCls(old_backbone, mode='fusion'))
    return backbone


def make_fusion_resnet_backbone(old_resnet: nn.Module,
                                new_resnet: nn.Module) -> nn.Module:
    return make_fused_backbone(old_resnet, new_resnet,
                               ResNetFeatureMapsExtractor,
                               (3, new_resnet.conv1.in_channels))


def make_fusion_effnet_backbone(old_resnet: nn.Module,
                                new_resnet: nn.Module) -> nn.Module:
    return make_fused_backbone(old_resnet, new_resnet,
                               EfficientNetFeatureMapsExtractor,
                               (3, new_resnet.conv1.in_channels))


def make_resnet_backbone(
        name: str, resnet: nn.Module, in_channels: int,
        pretrained: bool) -> Union[ResNetFeatureMapsExtractor, nn.Sequential]:
    if in_channels == 3:
        return ResNetFeatureMapsExtractor(resnet)

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
        return ResNetFeatureMapsExtractor(resnet)

    if in_channels > 3:
        new_channels = in_channels - 3
        new_conv = nn.Conv2d(in_channels=new_channels, **old_conv_args)

        resnet_cls = tv.models.resnet.__dict__[name]
        new_resnet = resnet_cls(pretrained=pretrained)
        new_resnet.conv1 = copy_conv_weights(old_conv, new_conv)

        backbone = make_fusion_resnet_backbone(resnet, new_resnet)
    else:
        # in_channels < 3
        new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
        resnet.conv1 = copy_conv_weights(old_conv, new_conv)
        backbone = ResNetFeatureMapsExtractor(resnet)
    return backbone
