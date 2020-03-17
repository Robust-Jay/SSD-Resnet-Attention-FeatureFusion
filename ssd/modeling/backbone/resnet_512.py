import torch.nn as nn
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

class Channel_Attention(nn.Module):

    def __init__(self, channel, r = 16):
        super(Channel_Attention, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)

        self._fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel // r, channel, 1, bias = False)
        )

        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self._avg_pool(x)
        y1 = self._fc(y1)

        y2 = self._max_pool(x)
        y2 = self._fc(y2)

        y = self._sigmoid(y1 + y2)
        return x * y


class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size = 3):
        super(Spatial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self._layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size = kernel_size, padding = padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim = 1, keepdim = True)
        max_mask, _= torch.max(x, dim = 1, keepdim = True)
        mask = torch.cat([avg_mask, max_mask], dim = 1)

        mask = self._layer(mask)
        return x * mask


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = dilation, groups = groups, bias = False, dilation = dilation)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1,stride = stride, bias = False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
                 base_width = 64, dilation = 1, norm_layer = None, reduction = 16, se = False, cbam = False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only support groups = 1 and base_width = 64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not support in BasicBlock")
        # Both self.conv1 and self.downsample layer downsample the input when stride != 1

        self.se = se
        self.cbam = cbam
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu =nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if self.se:
            self.se_layer = SELayer(planes, reduction)
        if self.cbam:
            self.ca = Channel_Attention(planes, reduction)
            self.sa = Spatial_Attention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se:
            out = self.se_layer(out)

        if self.cbam:
            out = self.ca(out)
            out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
                 base_width = 64, dilation = 1, norm_layer = None, reduction = 16, se = False, cbam = False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.se = se
        self.cbam = cbam
        self.conv1 = conv1x1(inplanes, width)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        if self.se:
            self.se_layer = SELayer(planes * self.expansion, reduction)
        if self.cbam:
            self.ca = Channel_Attention(planes * self.expansion, reduction)
            self.sa = Spatial_Attention()
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

        if self.se:
            out = self.se_layer(out)

        if self.cbam:
            out = self.ca(out)
            out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block = None, blocks = None, zero_init_residual = False,
                 groups=1, width_per_group=64, replace_stride_with_dilation = None,
                 norm_layer=None, extras = None, se = False, cbam = False, fusion = False):
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
        self.se = se
        self.cbam = cbam
        self.fusion = fusion
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, self.blocks[0])
        self.layer2 = self._make_layer(block, 128, self.blocks[1], stride = 2,
                                       dilate = replace_stride_with_dilation[0])
        if self.fusion:
            self._add_fusion_layer(512, 1024)
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
                            self.base_width, previous_dilation, norm_layer, se = self.se, cbam = self.cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, se = self.se, cbam = self.cbam))

        return nn.Sequential(*layers)

    def _add_fusion_layer(self, planes1, planes2):
        self.fu_conv1 = nn.Conv2d(planes1, planes1, kernel_size=3, stride=1, padding=1)
        self.fu_bn1 = nn.BatchNorm2d(planes1)
        self.fu_relu1 = nn.ReLU(inplace=True)

        self.fu_deconv = nn.ConvTranspose2d(planes2, planes1, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.fu_conv2 = nn.Conv2d(planes1, planes1, kernel_size=3, stride=1, padding=1)
        self.fu_bn2 = nn.BatchNorm2d(planes1)
        self.fu_relu2 = nn.ReLU(inplace=True)

        self.fu_conv3 = nn.Conv2d(planes2, planes1, kernel_size = 3, stride = 1, padding = 1)

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

        if self.fusion:
            x1 = x
            x2 = self.layer3(x1)
            x1 = self.fu_conv1(x1)
            x1 = self.fu_bn1(x1)
            x1 = self.fu_relu1(x1)

            x2 = self.fu_deconv(x2)
            x2 = self.fu_conv2(x2)
            x2 = self.fu_bn2(x2)
            x2 = self.fu_relu2(x2)

            y = torch.cat([x1, x2], dim=1)
            y = self.fu_conv3(y)
            features.append(y)
        else:
            features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        x = self.extra_layers[0](x)
        x = self.extra_layers[1](x)
        features.append(x)

        x = self.extra_layers[2](x)
        x = self.extra_layers[3](x)
        features.append(x)

        x = self.extra_layers[4](x)
        x = self.extra_layers[5](x)
        features.append(x)

        x = self.extra_layers[6](x)
        features.append(x)

        return tuple(features)


@registry.BACKBONES.register('Resnet18_512')
def Resnet18_512(cfg, pretrained=True):
    model = ResNet(BasicBlock, blocks=cfg.MODEL.RESNET.BLOCKS, extras=cfg.MODEL.RESNET.EXTRAS,
                   se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('Resnet34_512')
def Resnet34_512(cfg, pretrained=True):
    model = ResNet(BasicBlock, blocks=cfg.MODEL.RESNET.BLOCKS, extras=cfg.MODEL.RESNET.EXTRAS,
                   se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('Resnet50_512')
def Resnet50_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('Resnet101_512')
def Resnet101_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('Resnet152_512')
def Resnet152_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('Resnet50_32x4d_512')
def Resnet50_32x4d_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   groups = 32, width_per_group = 4, se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('Resnet101_32x8d_512')
def Resnet101_32x8d_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   groups = 32, width_per_group = 8, se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('wide_resnet50_2_512')
def wide_resnet50_2_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   width_per_group = 64 * 2, se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@registry.BACKBONES.register('wide_resnet101_2_512')
def wide_resnet101_2_512(cfg, pretrained=True):
    model = ResNet(Bottleneck, blocks = cfg.MODEL.RESNET.BLOCKS, extras = cfg.MODEL.RESNET.EXTRAS,
                   width_per_group = 64 * 2, se = cfg.MODEL.RESNET.SE, cbam = cfg.MODEL.RESNET.CBAM, fusion = cfg.MODEL.RESNET.FUSION)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['wide_resnet101_2'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    import torch
    from torchsummary import summary
    resnet = ResNet(block = Bottleneck, blocks = [3, 4, 6, 3], extras = [512, 256, 128, 64, 128], se = False, cbam = False, fusion = False)
    summary(resnet, (3, 512, 512))
    print(resnet)