"""
PyTorch implementation of Resnet, adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

# load packages
import os
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    _expansion = 1

    def __init__(self, in_planes, planes, stride=1, nl=nn.GELU()):
        super(BasicBlock, self).__init__()
        self.nl = nl
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self._expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self._expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self._expansion * planes),
            )

    def forward(self, x):
        out = self.nl(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.nl(out)
        return out


class Bottleneck(nn.Module):
    _expansion = 4

    def __init__(self, in_planes, planes, stride=1, nl=nn.GELU()):
        super(Bottleneck, self).__init__()
        self.nl = nl
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self._expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self._expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self._expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self._expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self._expansion * planes),
            )

    def forward(self, x):
        out = self.nl(self.bn1(self.conv1(x)))
        out = self.nl(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.nl(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, nl=nn.GELU()):
        super(ResNet, self).__init__()
        self.nl = nl
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block._expansion, num_classes)

        self.pooling = nn.AvgPool2d(4)

        # for geometric quantity computation purpose
        self.feature_map = nn.Sequential(
            self.conv1,
            self.bn1,
            self.nl,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.pooling,
            nn.Flatten(start_dim=1),  # take the place of view
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block._expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_map(x)
        out = self.linear(out)
        return out


def ResNet18(nl=nn.GELU()):
    return ResNet(BasicBlock, [2, 2, 2, 2], nl=nl)


def ResNet34(nl=nn.GELU()):
    return ResNet(BasicBlock, [3, 4, 6, 3], nl=nl)


def ResNet50(nl=nn.GELU()):
    return ResNet(Bottleneck, [3, 4, 6, 3], nl=nl)


def ResNet101(nl=nn.GELU()):
    return ResNet(Bottleneck, [3, 4, 23, 3], nl=nl)


def ResNet152(nl=nn.GELU()):
    return ResNet(Bottleneck, [3, 8, 36, 3])

def get_intermeidate_feature_maps(model: nn.Module):
    """
    get feature maps from intermediate layers
    
    Each resnet has four blocks. The feature map is defined by 
    appending avg pooling and flattening after each blocks
    """
    avg_kernel_size = 4 # ? what should be the most appropriate value ?
    model_list = model.feature_map

    # * align output dimension to be all 512
    yield nn.Sequential(*model_list[:4], nn.AvgPool2d((16, 8)), nn.Flatten(start_dim=1))
    yield nn.Sequential(*model_list[:5], nn.AvgPool2d(8), nn.Flatten(start_dim=1))
    yield nn.Sequential(*model_list[:6], nn.AvgPool2d((8, 4)), nn.Flatten(start_dim=1))
    yield nn.Sequential(*model_list[:7], nn.AvgPool2d(avg_kernel_size), nn.Flatten(start_dim=1))
