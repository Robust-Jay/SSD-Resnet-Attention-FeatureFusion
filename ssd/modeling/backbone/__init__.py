from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .resnet import *
from .resnet_512 import *

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet',
           'Resnet18', 'Resnet34', 'Resnet50','Resnet101', 'Resnet152',
           'Resnet50_32x4d', 'Resnet101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
           'Resnet18_512', 'Resnet34_512', 'Resnet50_512','Resnet101_512', 'Resnet152_512',
           'Resnet50_32x4d_512', 'Resnet101_32x8d_512', 'wide_resnet50_2_512', 'wide_resnet101_2_512']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
