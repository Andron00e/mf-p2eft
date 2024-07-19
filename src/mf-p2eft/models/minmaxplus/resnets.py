import torch
from torch import nn
import torch.nn.functional as F
import definitions.networks.modules as modules
from definitions.networks.modules import SolidConv


class TropicalBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, input, outputs):
        super(TropicalBasicBlock, self).__init__()
        self.l1 = modules.MinPlus(input, outputs)
        self.l2 = modules.Max_B_Plus(outputs, outputs)

        self.shortcut = torch.nn.Sequential()
        if input != outputs:
            self.shortcut = torch.nn.Sequential(
                modules.MinPlus(input, outputs), modules.Max_B_Plus(outputs, outputs)
            )

    def forward(self, x):
        out = self.l2(self.l1(x))
        out = torch.min(out, self.shortcut(x))
        return out


class TropcialBottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, input, outputs):
        super(TropcialBottleneck, self).__init__()
        self.layer1 = self._make_layer(TropicalBasicBlock, input, input)
        self.layer2 = self._make_layer(TropicalBasicBlock, input, input)
        self.layer3 = self._make_layer(TropicalBasicBlock, input, input)
        self.layer4 = self._make_layer(TropicalBasicBlock, input, outputs)

        self.shortcut = torch.nn.Sequential()
        if input != outputs:
            self.shortcut = torch.nn.Sequential(
                modules.MinPlus(input, outputs), modules.Max_B_Plus(outputs, outputs)
            )

    def _make_layer(self, block, input, outout):
        return torch.nn.Sequential(block(input, outout))

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out = self.layer4(self.layer3(out))
        out = torch.max(out, self.shortcut(x))
        return out


class ResNet(torch.nn.Module):
    def __init__(self, input, block, num_classes=10):
        super(ResNet, self).__init__()       
        self.layer1 = self._make_layer(block, input, 1024)
        self.layer2 = self._make_layer(block, 1024, 512)
        self.layer3 = modules.MinPlus(512, 256)
        self.layer4 = modules.Max_B_Plus(256, num_classes)

    def _make_layer(self, block, input, outout):
        return torch.nn.Sequential(block(input, outout))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def TropicalResNet(input, num_classes=10):
    return ResNet(input, TropcialBottleneck)


class ResNetConv(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetConv, self).__init__()
        self.in_planes = 64

        self.conv = nn.Sequential(
            SolidConv(3, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SolidConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SolidConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            modules.MinPlus(64 * 4 * 4, 64 * 2 * 2),
            modules.Max_B_Plus(64 * 2 * 2, 64)
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Sequential(        
            modules.MinPlus(512, 256),
            modules.Max_B_Plus(256, num_classes)
        )
        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)

        return x


def ResNet18():
    return ResNetConv(TropicalBasicBlock, [2, 2, 2, 2])