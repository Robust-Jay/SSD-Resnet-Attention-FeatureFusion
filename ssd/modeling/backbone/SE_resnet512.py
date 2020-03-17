import torch.nn as nn
import torch
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = dilation, groups = groups, bias = False, dilation = dilation)

def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
                 base_width = 64, dilation = 1, norm_layer = None, reduction = 16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SEBasicBlock only support groups = 1 and base_width = 64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not support in SEBasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inpanes, planes, stride =1, downsample = None, groups = 1,
                 base_width = 64, dilation = 1, norm_layer = None, reduction = 16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inpanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3= norm_layer(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEResNet(nn.Module):
    def __init__(self, block = None, blocks = None, zero_init_residual = False,
                 groups=1, width_per_group=64, replace_stride_with_dilation = None,
                 norm_layer=None, extras = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.blocks = blocks
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, self.blocks[0])
        self.layer2 = self._make_layer(block, 128, self.blocks[1], stride = 2,
                                       dilate = replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, self.blocks[2], stride = 2,
                                       dilate = replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, self.blocks[3], stride = 2,
                                       dilate = replace_stride_with_dilation[2])
        self.extra_layers = nn.Sequential(* self._add_extras(block, extras))

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _add_extras(self, block, extras):
        layers = []
        layers += self._make_layer(block, extras[1], 2, stride=2)
        layers += self._make_layer(block, extras[2], 2, stride=2)
        layers += self._make_layer(block, extras[3], 2, stride=2)
        in_channels = extras[3] * block.expansion
        layers += [nn.Conv2d(in_channels, extras[4] * block.expansion, kernel_size=2)]
        return layers

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        x = self.extra_layers[0](x)
        features.append(x)

        x = self.extra_layers[1](x)
        features.append(x)

        x = self.extra_layers[2](x)
        features.append(x)

        x = self.extra_layers[3](x)
        features.append(x)

        return tuple(features)


@registry.BACKBONES.register('SEResnet18_512')
def SEResnet18_512(cfg, pretrained=True):
    model = SEResNet(SEBasicBlock, blocks=cfg.MODEL.RESNET.BLOCKS, extras=cfg.MODEL.RESNET.EXTRAS)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEResnet34_512')
def SEResnet34_512(cfg, pretrained=True):
    model = SEResNet(SEBasicBlock, blocks=cfg.MODEL.RESNET.BLOCKS, extras=cfg.MODEL.RESNET.EXTRAS)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEResnet50_512')
def SEResnet50_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEResnet101_512')
def SEResnet101_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEResnet152_512')
def SEResnet152_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEResnet50_32x4d_512')
def SEResnet50_32x4d_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   groups = 32, width_per_group = 4)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEResnet101_32x8d_512')
def SEResnet101_32x8d_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   groups = 32, width_per_group = 8)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEwide_resnet50_2_512')
def SEwide_resnet50_2_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   width_per_group = 64 * 2)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('SEwide_resnet101_2_512')
def SEwide_resnet101_2_512(cfg, pretrained=True):
    model = SEResNet(SEBottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   width_per_group = 64 * 2)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['wide_resnet101_2'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model



if __name__ == '__main__':
    # net = SELayer(256)
    # x = torch.rand((32, 256, 300, 300))
    # net(x)
    import torch
    from torchsummary import summary

    resnet = SEResNet(block=SEBottleneck, blocks=[3, 4, 6, 3], extras=[512, 256, 128, 64])
    summary(resnet, (3, 300, 300))
    print(resnet)