import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Bottleneck(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        inter_ch = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(nn.ReLU()(self.bn1(x)))
        out = self.conv2(nn.ReLU()(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(nn.ReLU()(self.bn(x)))
        out = nn.AvgPool2d(kernel_size=(2, 2), stride=2)(out)
        return out


class Transition_for_Decorder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(nn.ReLU()(self.bn(x)))
        out = nn.Upsample(scale_factor=2, mode='nearest')(out)
        return out


class Densenet(nn.Module):
    def __init__(self, bottleneck = Bottleneck, nblocks = [6, 12, 24, 16], growth_rate=12, reduction=0.5, num_classes=10):
        # DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)
        super().__init__()
        self.growth_rate = growth_rate  # 12

        in_ch = 2 * growth_rate  # 24
        self.conv1 = nn.Conv2d(3, in_ch, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(bottleneck, in_ch, nblocks[0])
        in_ch += nblocks[0] * growth_rate  # denselayerの出力ch: 24 + 6*12 = 96
        out_ch = int(math.floor(in_ch * reduction))  # 96*0.5 = 48
        self.trans1 = Transition(in_ch, out_ch)  # 48, 16, 16

        in_ch = out_ch  # 48
        self.dense2 = self._make_dense_layers(bottleneck, in_ch, nblocks[1])
        in_ch += nblocks[1] * growth_rate
        out_ch = int(math.floor(in_ch * reduction))
        self.trans2 = Transition(in_ch, out_ch)  # 96, 8, 8

        in_ch = out_ch  # 96
        self.dense3 = self._make_dense_layers(bottleneck, in_ch, nblocks[2])
        in_ch += nblocks[2] * growth_rate
        out_ch = int(math.floor(in_ch * reduction))
        self.trans3 = Transition(in_ch, out_ch)  # 192, 4, 4

        in_ch = out_ch # 192
        self.dense4 = self._make_dense_layers(bottleneck, in_ch, nblocks[3])
        in_ch += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(in_ch)
        self.linear = nn.Linear(in_ch, num_classes)

    def _make_dense_layers(self, bottleneck, in_ch, nblock):
        layers = []
        for i in range(nblock):
            layers.append(bottleneck(in_ch, self.growth_rate))
            in_ch += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))  # 48, 16, 16
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)  # 384,1, 1
        # out = F.avg_pool2d(F.relu(self.bn(out)), 8) #384,1, 1
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

"""
class Densenet_det(nn.Module):
    def __init__(self, bottleneck, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate  # 12

        in_ch = orig_ch = 192
        self.dense1 = self._make_dense_layers(bottleneck, in_ch, nblocks[0])
        in_ch += nblocks[0] * growth_rate
        self.bn = nn.BatchNorm2d(in_ch)
        self.linear = nn.Linear(in_ch, num_classes)

    def _make_dense_layers(self, bottleneck, in_ch, nblock):
        layers = []
        for i in range(nblock):
            layers.append(bottleneck(in_ch, self.growth_rate))
            in_ch += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense1(x)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)  # 384,1, 1
        # out = F.avg_pool2d(F.relu(self.bn(out)), 8) #384,1, 1
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        # out = nn.Dropout2d(p=0.5)(out)
        return out


def enc_modules():
    return Densenet(Bottleneck, [6, 12, 24], growth_rate=12)


def dec_modules():
    return Densenet_dec(Bottleneck, [24, 12, 6], growth_rate=12)


def det_modules():
    return Densenet_det(Bottleneck, [16], growth_rate=12)


class Densenet_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = enc_modules()
        self.det = det_modules()

    def forward(self, x):
        # enc module forward
        enc = self.enc(x)

        # dec module forward
        det = self.det(enc)
        return enc, det
"""