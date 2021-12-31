import unittest

import torch

from fpn.factory import make_fpn_resnet, make_fpn_efficientnet

DEVICE = 'cuda:0'

NUM_CLASSES = 2
IMG_SIZE = 224
FPN_CHANNELS = 64


class TestFPN(unittest.TestCase):
    def _test_fpn(self,
                  factory_fn,
                  fpn_type,
                  backbones,
                  in_channels=[3, 1, 8],
                  pretrained=True,
                  fpn_channels=FPN_CHANNELS,
                  num_classes=NUM_CLASSES,
                  size=(IMG_SIZE, IMG_SIZE)):
        for backbone in backbones:
            for ch in in_channels:
                test_name = f'{fpn_type}, {backbone}, {ch}-channel input'
                with self.subTest(msg=test_name):
                    model = factory_fn(
                        name=backbone,
                        fpn_type=fpn_type,
                        pretrained=pretrained,
                        num_classes=NUM_CLASSES,
                        fpn_channels=fpn_channels,
                        in_channels=ch,
                        out_size=size).to(DEVICE)
                    x = torch.empty((1, ch, *size)).to(DEVICE)
                    out = model(x)
                    self.assertEqual(out.shape, (1, num_classes, *size))
                    del model
                    del x
                    del out
                    if DEVICE.startswith('cuda'):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

    def test_fpn_resnet(self):
        resnets = ['resnet18', 'resnet50', 'resnet101']
        self._test_fpn(
            factory_fn=make_fpn_resnet, fpn_type='fpn', backbones=resnets)

    def test_panoptic_fpn_resnet(self):
        resnets = ['resnet18', 'resnet50', 'resnet101']
        self._test_fpn(
            factory_fn=make_fpn_resnet, fpn_type='panoptic', backbones=resnets)

    def test_panet_fpn_resnet(self):
        resnets = ['resnet18', 'resnet50', 'resnet101']
        self._test_fpn(
            factory_fn=make_fpn_resnet, fpn_type='panet', backbones=resnets)

    def test_fpn_efficientnet(self):
        effnets = [f'efficientnet_b{i}' for i in range(8)]
        self._test_fpn(
            factory_fn=make_fpn_efficientnet,
            fpn_type='fpn',
            backbones=effnets)

    def test_panoptic_fpn_efficientnet(self):
        effnets = [f'efficientnet_b{i}' for i in range(8)]
        self._test_fpn(
            factory_fn=make_fpn_efficientnet,
            fpn_type='panoptic',
            backbones=effnets)

    def test_panet_fpn_efficientnet(self):
        effnets = [f'efficientnet_b{i}' for i in range(8)]
        self._test_fpn(
            factory_fn=make_fpn_efficientnet,
            fpn_type='panet',
            backbones=effnets)


if __name__ == '__main__':
    unittest.main(failfast=True)
