from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs):
        if isinstance(xs, torch.Tensor):
            return tuple(m(xs) for m in self)
        return tuple(m(x) for m, x in zip(self, xs))


class SequentialMultiInputOutput(nn.Sequential):
    def forward(self, inps):
        outputs = [None] * len(self)
        out = self[0](inps[0])
        outputs[0] = out
        for i, (module, inp) in enumerate(zip(self[1:], inps[1:]), start=1):
            out = module((inp, out))
            outputs[i] = out
        return outputs


class Reverse(nn.Module):
    def forward(self, inps):
        return inps[::-1]


class Interpolate(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fn = partial(F.interpolate, **kwargs)

    def forward(self, x):
        return self.fn(x)


class AddTensors(nn.Module):
    def forward(self, inps):
        return sum(inps)


class Residual(nn.Sequential):
    def __init__(self, layer):
        super().__init__(
            Parallel([nn.Identity(), layer]),
            AddTensors()
        )


class SelectOne(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        return xs[self.idx]


class FPN(nn.Sequential):
    def __init__(self, in_feats_shapes: list, num_channels: int = 256):
        in_convs = Parallel([
            nn.Conv2d(s[1], num_channels, kernel_size=1)
            for s in in_feats_shapes[::-1]
        ])
        upsample_and_add = SequentialMultiInputOutput(*[
            Residual(
                Interpolate(size=s[-2:], mode='bilinear', align_corners=True))
            for s in in_feats_shapes[::-1]
        ])
        out_convs = Parallel([
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            for s in in_feats_shapes[::-1]
        ])
        layers = [
            Reverse(),
            in_convs,
            upsample_and_add,
            out_convs,
            Reverse()
        ]
        super().__init__(*layers)


class PANetFPN(nn.Sequential):
    def __init__(self, in_feats_shapes: list, num_channels: int = 256):
        fpn1 = FPN(in_feats_shapes, num_channels=num_channels)
        fpn2 = FPN(in_feats_shapes[::-1], num_channels=num_channels)
        layers = [
            fpn1,
            Reverse(),
            fpn2
        ]
        super().__init__(*layers)


def _get_shapes(m, sz=224):
    state = m.training
    m.eval()
    with torch.no_grad():
        feats = m.extract_endpoints(torch.empty(1, 3, sz, sz))
    m.train(state)
    return [f.shape for f in feats.values()]


class EfficientNetFeatureMapsExtractor(nn.Module):
    def __init__(self, effnet):
        super().__init__()
        self.m = effnet

    def forward(self, x):
        feats = self.m.extract_endpoints(x)
        return list(feats.values())


def _load_efficientnet(name,
                       num_classes=1000,
                       pretrained='imagenet',
                       in_channels=3):
    model = torch.hub.load(
        'lukemelas/EfficientNet-PyTorch',
        name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels
    )
    return model


def make_segm_fpn_efficientnet(name='efficientnet_b0',
                               out_size=(224, 224),
                               fpn_channels=256,
                               num_classes=1000,
                               pretrained='imagenet',
                               in_channels=3):
    effnet = _load_efficientnet(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )
    model = nn.Sequential(
        EfficientNetFeatureMapsExtractor(effnet),
        FPN(_get_shapes(effnet), num_channels=fpn_channels),
        SelectOne(idx=0),
        Interpolate(size=out_size, mode='bilinear', align_corners=True)
    )
    return model
