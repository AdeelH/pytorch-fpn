# pytorch-fpn

PyTorch implementation of the following semantic segmentation architectures (see `fpn.py`):
- **FPN**, *Feature Pyramid Networks for Object Detection* by Lin et al., https://arxiv.com/abs/1612.03144.
- **Panoptic FPN**, *Panoptic Feature Pyramid Networks* by Kirilov et al., https://arxiv.com/abs/1901.02446.
- **PANet FPN**, *Path Aggregation Network for Instance Segmentation* by Liu et al., https://arxiv.com/abs/1803.01534

The implementations are all based on `nn.Sequential` with no fancy forward methods, meaning that they can be easily modified and combined together or with other modules.


# Backbones
The repo provides factory functions (`make_fpn_resnet()` and `make_fpn_efficientnet()`) for creating FPN's with 2 kinds of backbones:
- ResNet backbones (via Torch Vision)
- EfficientNet backbones (via https://github.com/lukemelas/EfficientNet-PyTorch)


# Multiband images
The factory methods `make_fpn_resnet()` and `make_fpn_efficientnet()` support `in_channels != 3`.

`make_fpn_resnet()`, in particular, makes use of the fusion technique described in the paper, *FuseNet*, by Hazirbas et al. (https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf) if `in_channels > 3` that adds a parallel resnet backbone for the new channels. All the pretrained weights are retained.

# Usage

```python
from factory import make_fpn_resnet

model = make_fpn_resnet(
	name='resnet18',
	fpn_type='fpn',
	pretrained=True,
	num_classes=2,
	fpn_channels=256,
	in_channels=3,
	out_size=(224, 224))

```

# Loading through torch.hub
This repo supports importing modules through `torch.hub`. The models can be easily imported into your code via the factory functions in `factory.py`.

```python

import torch

model = torch.hub.load(
	'AdeelH/pytorch-fpn',
	'make_fpn_resnet',
	name='resnet18',
	fpn_type='panoptic',
	num_classes=2,
	fpn_channels=256,
	in_channels=3,
	out_size=(224, 224)
)

```

Working example on Colab: https://colab.research.google.com/drive/1pjkw-QoqiXgKDrEYZ47EwfXcW7vgO4KX?usp=sharing
