from typing import Tuple, Optional, Type

from torch import nn

from containers import (Parallel, SequentialMultiInputMultiOutput,
                        SequentialMultiOutput)
from layers import (Sum, SplitTensor)


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
        # yapf: enable
        if mode == 'fusion':
            self.m = nn.Sequential(
                Parallel([stem, nn.Identity()]),
                SequentialMultiInputMultiOutput(
                    *[nn.Sequential(Sum(), m) for m in layers]))
        else:
            self.m = SequentialMultiOutput(stem, *layers)

    def forward(self, x):
        if self.mode != 'fusion':
            return self.m(x)
        x, inps = x
        return self.m((x, inps))


def make_fused_backbone(old_backbone: nn.Module, new_backbone: nn.Module,
                        featureMapExtractorCls: Type,
                        channel_split: Tuple[int, int]) -> nn.Module:
    """Create a fused backbone using FuseNet style feature fusion.
    See the paper, "FuseNet", by Hazirbas et al. for details.
    """
    backbone = nn.Sequential(
        SplitTensor(size_or_sizes=channel_split, dim=1),
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
