import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class Conv_Bn_Activation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation="linear", conv = "Conv2d", groups=1, dilation=1, pool=1, bn=True, bias=False, UpSample=1, dropout=0):
        super().__init__()
        self.conv = nn.ModuleList()
        pad = (kernel_size - 1) // 2
        
        #if UpSample != 1:
        #    self.conv.append(nn.Upsample(scale_factor=UpSample, mode = 'nearest'))      
        
        if conv == "Conv2d":
            if bias:
                self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, groups=groups, dilation=dilation))
            else:
                self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, groups=groups, dilation=dilation, bias=False))                                
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels, track_running_stats=False))            
        if activation == "tanh":
            self.conv.append(nn.Tanh())
        elif activation == "sigmoid":
            self.conv.append(nn.Sigmoid())
        elif activation == "mish":
            self.conv.append(Mish())
        elif activation == "swish":
            self.conv.append(Swish())            
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!!")
        
        if dropout != 0:
            self.conv.append(nn.Dropout2d(p=dropout))
        if pool != 1:
            self.conv.append(nn.MaxPool2d(pool, pool))
            
    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class SEblock(nn.Module): # Squeeze Excitation
    def __init__(self, ch_in, ch_sq):
        super().__init__()
        self.module_list = nn.ModuleList()

        self.module_list.append(nn.AdaptiveAvgPool2d(1))
        self.module_list.append(nn.Conv2d(ch_in, ch_sq, 1))
        self.module_list.append(Swish())
        self.module_list.append(nn.Conv2d(ch_sq, ch_in, 1))

    def forward(self, x):
        orig_x = x
        for module in self.module_list:
            x = module(x)
        return orig_x * torch.sigmoid(x)

class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0),1,1,1], dtype=x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x

class BMConvBlock(nn.Module): #32, 16, 1, 1, 3
    def __init__(self, ch_in, ch_out, expand_ratio, stride, kernel_size, reduction_ratio=4, drop_connect_rate=0.2):
        super().__init__()
        self.use_residual = (ch_in == ch_out) & (stride == 1)
        ch_med = int(ch_in * expand_ratio) #32*1 = 32
        ch_sq = max(1, ch_in // reduction_ratio) #32/4 = 8

        #print("kernel_size", kernel_size, "stride", stride)
        self.module_list = nn.ModuleList()
        if expand_ratio != 1.0:
            self.module_list.append(Conv_Bn_Activation(ch_in, ch_med, kernel_size=1, stride=1, activation = 'swish', groups=1))

        self.module_list.append(Conv_Bn_Activation(ch_med, ch_med, kernel_size=kernel_size, stride=stride, activation = 'swish', groups=ch_med))
        self.module_list.append(SEblock(ch_med, ch_sq))
        self.module_list.append(Conv_Bn_Activation(ch_med, ch_out, kernel_size=1, stride=1, activation = 'linear', groups=1))

        if self.use_residual:
            self.drop_connect = DropConnect(drop_connect_rate)

    def forward(self, x):
        orig_x = x
        for module in self.module_list:
                x = module(x)
        if self.use_residual:
            return orig_x + self.drop_connect(x)
        else:
            return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, resolution=False, dropout_rate=0.2, ch_in=3, num_classes=1000):
        super().__init__()

        # expand_ratio, channel, repeats, stride, kernel_size
        settings = [
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7]
        ]

        ch_out = int(math.ceil(32*width_mult)) #32*1 = 32
        self.features = nn.ModuleList()
        self.features.append(Conv_Bn_Activation(ch_in, ch_out, kernel_size=3, stride=2, activation = 'swish', groups=1))

        ch_in = ch_out
        for t, c, n, s, k in settings:
            ch_out  = int(math.ceil(c*width_mult))
            repeats = int(math.ceil(n*depth_mult))
            for i in range(repeats):
                stride = s if i==0 else 1 #stride=1
                self.features.append(BMConvBlock(ch_in, ch_out, t, stride, k))
                ch_in = ch_out
        ch_last = int(math.ceil(1280*width_mult))
        self.features.append(Conv_Bn_Activation(ch_in, ch_last, kernel_size=1, stride=1, activation = 'swish', groups=1))

        self.classifier = nn.ModuleList()
        self.classifier.append(nn.AdaptiveAvgPool2d(1))
        self.classifier.append(Flatten())
        self.classifier.append(nn.Dropout(dropout_rate))
        self.classifier.append(nn.Linear(ch_last, num_classes))

    def forward(self, x):
        for module in self.features:
            x = module(x)
        for module in self.classifier:
            x = module(x)
        return x

def efficientnet_b0(input_ch=3, num_classes=10):
    #(w_mult, d_mult, resolution, droprate) = (1.0, 1.0, 224, 0.2)
    return EfficientNet(1.0, 1.0, None, 0.2, input_ch, num_classes)

def efficientnet_b1(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.0, 1.1, 240, 0.2)
    return EfficientNet(1.0, 1.1, None, 0.2, input_ch, num_classes)

def efficientnet_b2(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.1, 1.2, 260, 0.3)
    return EfficientNet(1.1, 1.2, None, 0.3, input_ch, num_classes)

def efficientnet_b3(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.2, 1.4, 300, 0.3)
    return EfficientNet(1.2, 1.4, None, 0.3, input_ch, num_classes)

def efficientnet_b4(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.4, 1.8, 380, 0.4)
    return EfficientNet(1.4, 1.8, None, 0.4, input_ch, num_classes)

def efficientnet_b5(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.6, 2.2, 456, 0.4)
    return EfficientNet(1.6, 2.2, None, 0.4, input_ch, num_classes)

def efficientnet_b6(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.8, 2.6, 528, 0.5)
    return EfficientNet(1.8, 2.6, None, 0.5, input_ch, num_classes)

def efficientnet_b7(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (2.0, 3.1, 600, 0.5)
    return EfficientNet(2.0, 3.1, None, 0.5, input_ch, num_classes)

