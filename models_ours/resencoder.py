# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_ours import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

def conv_1(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, bias=bias)
'''
class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res'''
class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResEncoder_src(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(ResEncoder_src, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        self.out_dim = args.out_dim
        kernel_size = 3
        act = nn.ReLU(True)
        self.conv_start = conv_1
        kernelsize_start = 1
        # define head module
        m_head = [self.conv_start(args.n_colors, n_feats, kernelsize_start)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        m_tail = [conv(n_feats, self.out_dim, kernel_size)]
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        # check input and output nan
        if torch.isnan(x).any():
            print("Input contains NaN values")
        x = self.head(x)
        if torch.isnan(x).any():
            print("After head, contains NaN values")
        res = self.body(x)
        if torch.isnan(res).any():
            print("After body, contains NaN values")
        res += x
        if torch.isnan(res).any():
            print("After adding residual, contains NaN values")
        x = self.tail(res)
        if torch.isnan(x).any():
            print("After tail, contains NaN values")

        return x
        
class ResEncoder_tgt(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(ResEncoder_tgt, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        self.out_dim = args.out_dim
        kernel_size = 3
        act = nn.ReLU(True)
        self.conv_end = conv_1
        kernelsize_end = 1
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        m_tail = [self.conv_end(n_feats, self.out_dim, kernelsize_end)]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x

@register('encoder_src')
def make_encoder_src(n_resblocks=24, n_feats=1024, outdim=1024,
              res_scale=0.1, scale=1, input_channels=8):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.out_dim = outdim
    args.res_scale = res_scale
    args.scale = [scale]
    args.n_colors = input_channels
    return ResEncoder_src(args)

@register('encoder_tgt')
def make_encoder_src(n_resblocks=3, n_feats=1024, outdim=8,
              res_scale=1, scale=1, input_channels=8):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.out_dim = outdim
    args.res_scale = res_scale
    args.scale = [scale]
    args.n_colors = input_channels
    return ResEncoder_tgt(args)

