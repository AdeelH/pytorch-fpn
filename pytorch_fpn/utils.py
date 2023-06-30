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
    # save state so we can restore later
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


def download_from_s3(uri: str, download_path: str) -> None:
    """Download a file from S3."""
    try:
        import boto3
    except ImportError as e:
        e.msg + ' (boto3 is required for downloading files from S3)'
        raise e

    from urllib.parse import urlparse

    s3 = boto3.Session().client('s3')
    request_payer = 'requester'
    parsed_uri = urlparse(uri)
    bucket, key = parsed_uri.netloc, parsed_uri.path[1:]

    file_size = s3.head_object(Bucket=bucket, Key=key)['ContentLength']

    print(f'Downloading {uri} to {download_path}...')

    try:
        from tqdm.auto import tqdm
        progressbar = tqdm(
            total=file_size,
            desc='Downloading',
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            mininterval=0.5)
        with progressbar as bar:
            s3.download_file(
                Bucket=bucket,
                Key=key,
                Filename=download_path,
                Callback=lambda bytes: bar.update(bytes),
                ExtraArgs={'RequestPayer': request_payer})
    except ImportError:
        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=download_path,
            ExtraArgs={'RequestPayer': request_payer})


def load_state_dict_from_s3(uri: str) -> dict:
    from os.path import join
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        download_path = join(tmp_dir, 'backbone_weights.pth')
        download_from_s3(uri, download_path)
        state_dict = torch.load(download_path)
    return state_dict


def load_state_dict(uri: str) -> dict:
    if uri.startswith('s3://'):
        state_dict = load_state_dict_from_s3(uri)
    elif (uri.startswith('http') or uri.startswith('ftp')):
        state_dict = torch.hub.load_state_dict_from_url(uri)
    else:
        state_dict = torch.load(uri)
    return state_dict


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


class FrozenModule(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m
        self.m.eval()
        freeze(self.m)

    def train(self, mode: bool = True) -> 'FrozenModule':
        """Do nothing."""
        return self

    def forward(self, *args, **kwargs):
        return self.m(*args, **kwargs)
