from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .resnet_input_300 import *
from .resnet_input_512 import *

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet',
            'R50_300', 'R101_300',
           'R50_512', 'R101_512',
           ]

def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
