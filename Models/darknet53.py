import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class resblock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch//2, kernel_size=1, stride=1, activation = 'leaky'))
            resblock_one.append(Conv_Bn_Activation(ch//2, ch, kernel_size=3, stride=1, activation = 'leaky'))
            self.module_list.append(resblock_one)  
            
    def forward(self,x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h          
        return x


class Conv_Bn_Activation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, conv = "Conv2d", dilation=1, pool=1, bn=True, bias=False, UpSample=1):
        super().__init__()
        self.conv = nn.ModuleList()
        pad = (kernel_size - 1) // 2
        
        if UpSample != 1:
            self.conv.append(nn.Upsample(scale_factor=UpSample, mode = 'nearest'))        
        if conv == "Conv2d":
            if bias:
                self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation))
            else:
                self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation, bias=False))                                
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels, track_running_stats=False))            
        if activation == "tanh":
            self.conv.append(nn.Tanh())
        elif activation == "sigmoid":
            self.conv.append(nn.Sigmoid())
        elif activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            #self.conv.append(nn.LeakyReLU(0.1, inplace=True))
            self.conv.append(nn.LeakyReLU(0.1))
        elif activation == "linear":
            pass
        else:
            print("activate error !!!")
        if pool != 1:
            self.conv.append(nn.MaxPool2d(pool, pool))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x
    
def darknet53_modules():
    module_list = nn.ModuleList()
    module_list.append(Conv_Bn_Activation(3, 32, kernel_size = 3, stride = 1, activation = 'leaky')) #32, 32, 32
    module_list.append(Conv_Bn_Activation(32, 64, kernel_size = 3, stride = 2, activation = 'leaky')) #64, 16, 16
    module_list.append(resblock(ch=64)) #64, 16, 16
    module_list.append(Conv_Bn_Activation(64, 128, kernel_size = 3, stride = 2, activation = 'leaky')) #128, 8, 8
    module_list.append(resblock(ch=128, nblocks=2)) #128, 8, 8
    module_list.append(Conv_Bn_Activation(128, 256, kernel_size=3, stride=2, activation = 'leaky')) #256, 4, 4
    module_list.append(resblock(ch=256, nblocks=8))    #256, 4, 4
    module_list.append(Conv_Bn_Activation(256, 512, kernel_size=3, stride=2, activation = 'leaky')) #512, 2, 2
    module_list.append(resblock(ch=512, nblocks=8))    #512, 2, 2
    module_list.append(Conv_Bn_Activation(512, 1024, kernel_size=3, stride=2, activation = 'leaky'))#1024, 1, 1
    module_list.append(resblock(ch=1024, nblocks=4))#1024, 1, 1
    module_list.append(Conv_Bn_Activation(1024, 10, kernel_size=1, stride=1, activation = 'leaky'))#10, 1, 1
    return module_list

class darknet53(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = darknet53_modules()
        
    def forward(self, x):
        #enc module forward
        for module in self.module_list:
            x = module(x)
        x = x.reshape(-1, 10)
        return x
