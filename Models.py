# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from module import ESAM
import math
from thop import profile
from torchstat import stat


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



class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.fca = fcaLayer(dim)

    def forward(self, x, y):
        B, C, H, W = x.shape        #16,512,8,8
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        y = torch.fft.rfft2(y, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x1 = x * weight
        y1 = y * weight
        x2 = x1 + y
        y2 = y1 + x
        
        x2real = x2.real
        x2imag = x2.imag
        x2real = self.fca(x2real)
        x2imag = self.fca(x2imag)
        x = torch.stack([x2real, x2imag], dim=-1)
        x2 = torch.view_as_complex(x)
        
        y2real = y2.real
        y2imag = y2.imag
        y2real = self.eca(y2real)
        y2imag = self.eca(y2imag)
        y = torch.stack([y2real, y2imag], dim=-1)
        y2 = torch.view_as_complex(y)
        
        x = torch.fft.irfft2(x2, s=(H, W), dim=(1, 2), norm='ortho')
        y = torch.fft.irfft2(y2, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        return x, y

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class fcaLayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(fcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Fusion(nn.Module):
    def __init__(self, inter_channels, out_channels, h, w):
        super(Fusion, self).__init__()
        self.Global = GlobalFilter(out_channels, h, w)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.conv5a = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(out_channels))

        self.conv51 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(out_channels))

        self.conv11 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(out_channels))

    def forward(self, x, y):
        featx = self.conv5a(x)
        featy = self.conv5a(y)
        sx_feat, sy_feat = self.Global(featx, featy)
        sx_feat = sx_feat + featx
        sy_feat = sy_feat + featy
        sx_conv = self.conv51(sx_feat)
        sy_conv = self.conv51(sy_feat)


        return sx_conv, sy_conv

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

class Downsample(nn.Module):
    def __init__(self, inter_channels, out_channels):
        super(Downsample, self).__init__()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2d = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
            )
        self.conv1 = nn.Conv2d(inter_channels*3, out_channels,  kernel_size=1, stride=1, padding=0, bias=True)
        self.se = SqEx(3*inter_channels)
        self.fca = fcaLayer(3 * inter_channels)
    def forward(self, x):
        max = self.max1(x)
        avg = self.avg1(x)
        conv = self.conv2d(x)
        fusion = torch.cat((max, avg, conv), dim=1)
        eca = self.fca(fusion)
        conv1 = self.conv1(eca)

        return conv1

class Downsample1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))

        return y


class my_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(my_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = Downsample(32, 32)
        self.Maxpool2 = Downsample(64, 64)
        self.Maxpool3 = Downsample(128, 128)
        self.Maxpool4 = Downsample(256, 256)

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

        self.down1 = Downsample1(img_ch, 32)
        self.down2 = Downsample1(32, 64)
        self.down3 = Downsample1(64, 128)
        self.down4 = Downsample1(128, 256)
        self.down5 = Downsample1(256, 512)

        self.edge = ESAM(1)
        self.edge1 = ESAM(32)
        self.edge2 = ESAM(64)
        self.edge3 = ESAM(128)
        self.edge4 = ESAM(256)
        self.edge5 = ESAM(512)

        self.fusion1 = Fusion(32, 32, 256, 129)
        self.fusion2 = Fusion(64, 64, 128, 65)
        self.fusion3 = Fusion(128, 128, 64, 33)
        self.fusion4 = Fusion(256, 256, 32, 17)
        self.fusion5 = Fusion(512, 512, 16, 9)

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
        e1 = self.down1(x)
        edge1 = self.edge1(e1)
        x1 = x1 + edge1

        fx1, fy1 = self.fusion1(x1, y1)

        x2 = self.Maxpool1(fx1)
        x2 = self.Conv21(x2)
        e2 = self.Maxpool1(e1)
        e2 = self.down2(e2)

        edge2 = self.edge2(e2)
        x2 = x2 + edge2

        y2 = self.Maxpool1(fy1)
        y2 = self.Conv22(y2)

        fx2, fy2 = self.fusion2(x2, y2)

        x3 = self.Maxpool2(fx2)
        x3 = self.Conv31(x3)
        e3 = self.Maxpool2(e2)
        e3 = self.down3(e3)

        edge3 = self.edge3(e3)
        x3 = x3 + edge3


        y3 = self.Maxpool2(fy2)
        y3 = self.Conv32(y3)

        fx3, fy3 = self.fusion3(x3, y3)

        x4 = self.Maxpool3(fx3)
        x4 = self.Conv41(x4)

        e4 = self.Maxpool3(e3)
        e4 = self.down4(e4)
        edge4 = self.edge4(e4)
        x4 = x4 + edge4

        y4 = self.Maxpool3(fy3)
        y4 = self.Conv42(y4)

        fx4, fy4 = self.fusion4(x4, y4)

        x5 = self.Maxpool4(fx4)
        x5 = self.Conv51(x5)

        e5 = self.Maxpool4(e4)
        e5 = self.down5(e5)

        edge5 = self.edge5(e5)
        x5 = x5 + edge5

        y5 = self.Maxpool4(fy4)
        y5 = self.Conv52(y5)

        fx5, fy5 = self.fusion5(x5, y5)

        f5 = fx5 + fy5
        d5 = self.Up5(f5)
        d5 = torch.cat((fx4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((fx3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((fx2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((fx1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.sigmoid(d1)

        out2 = F.interpolate(edge2, scale_factor=2, mode='bilinear', align_corners=True)

        out3 = F.interpolate(edge3, scale_factor=4, mode='bilinear', align_corners=True)

        out4 = F.interpolate(edge4, scale_factor=8, mode='bilinear', align_corners=True)

        out = edge1 + out2
        out = self.edge(out)

        out = out + out3
        out = self.edge(out)

        out = out + out4
        out = self.edge(out)
        out = F.sigmoid(out)

        return d1, out



if __name__ == '__main__':
    images = torch.rand(1, 1, 256, 256)
    pet = torch.rand(1, 1, 256, 256)
    model = my_Net()
    s0 = model(images, pet)
    flops, param = profile(model, (images, pet,))
    print(fr'Param:{param}')
    print(fr'FLOPs:{flops}')
    print('flops: %.2f Gbps,params: %.2f M' % (flops / 1e9, param / 1e6))
