# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear = False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x

class fusion(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class my_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(my_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fusion1 = fusion(32, 32)
        self.fusion2 = fusion(64, 64)
        self.fusion3 = fusion(128, 128)
        self.fusion4 = fusion(256, 256)
        self.fusion5 = fusion(512, 512)

        self.Conv11 = conv_block(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv21 = conv_block(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv31 = conv_block(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv41 = conv_block(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv51 = conv_block(ch_in=filters_number[3], ch_out=filters_number[4])

        self.Conv12 = conv_block(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv22 = conv_block(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv32 = conv_block(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv42 = conv_block(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv52 = conv_block(ch_in=filters_number[3], ch_out=filters_number[4])


        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv5 = conv_block(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_block(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_block(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_block(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):

        x1 = self.Conv11(x)
        y1 = self.Conv12(y)

        f1 = self.fusion1(x1+y1)

        x1 = x1+f1
        x2 = self.Maxpool(x1)
        x2 = self.Conv21(x2)

        y1 = y1+f1
        y2 = self.Maxpool(y1)
        y2 = self.Conv22(y2)

        f2 = self.fusion2(x2 + y2)

        x2 = f2 + x2
        x3 = self.Maxpool(x2)
        x3 = self.Conv31(x3)

        y2 = f2 +y2
        y3 = self.Maxpool(y2)
        y3 = self.Conv32(y3)

        f3 = self.fusion3(x3 + y3)

        x3 = f3 + x3
        x4 = self.Maxpool(x3)
        x4 = self.Conv41(x4)

        y3 = f3 + y3
        y4 = self.Maxpool(y3)
        y4 = self.Conv42(y4)

        f4 = self.fusion4(x4 + y4)

        x4 = f4 + x4
        x5 = self.Maxpool(x4)
        x5 = self.Conv51(x5)

        y4 = f4 + y4
        y5 = self.Maxpool(y4)
        y5 = self.Conv52(y5)

        f5 = self.fusion5(x5 + y5)

        d5 = self.Up5(f5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
