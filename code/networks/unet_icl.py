# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import pdb
from collections import OrderedDict
from collections.abc import Iterable
from typing import Sequence, Tuple, Type, Union, Any

import numpy as np
import torch
import torch.nn as nn
from monai.networks.layers import Conv, DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from torch.distributions.uniform import Uniform
from torch.nn import LayerNorm



def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x_1 = self.up1(x4, x3) 
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x = self.up4(x_3, x0)
        feats = [x_1, x_2, x_3]

        output = self.out_conv(x)

        return output, feats

class UNet_icl(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_icl, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'input_resolution':[16, 32, 64, 128, 256],
                  'num_heads':(2, 4, 8),
                  'depths':(2, 2, 2),
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        icl_in_chans = (params['feature_chns'][3], params['feature_chns'][2], params['feature_chns'][1])
        icl_in_resolutions = (params['input_resolution'][1], params['input_resolution'][2], params['input_resolution'][3])
        icl_num_heads = params['num_heads'][::-1]
        patch_size = ensure_tuple_rep(2, 2)

        self.sspa = InherentConsistent(
            in_chans=icl_in_chans, # different in_chans for different scales of feature maps
            depths=params['depths'],
            patch_size=patch_size, 
            input_resolution=icl_in_resolutions,
            num_classes=class_num,
            num_heads=icl_num_heads,
            norm_layer=nn.LayerNorm,
        )
        self.uscl = InherentConsistent(
            in_chans=icl_in_chans, # different in_chans for different scales of feature maps
            depths=params['depths'],
            patch_size=patch_size, 
            input_resolution=icl_in_resolutions,
            num_classes=class_num,
            num_heads=icl_num_heads,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x_lab, x_unlab=None, inference=False):
        feature_lab = self.encoder(x_lab)
        output_lab, feats_lab = self.decoder(feature_lab)

        if inference:
            return output_lab
        else:
            feature_unlab = self.encoder(x_unlab)
            output_unlab, feats_unlab = self.decoder(feature_unlab)
            
            feat_Maps_lab, updated_Qs_lab = self.sspa(feats_lab, 'labeled')
            feat_Maps_consisunlab, _ = self.sspa(feats_unlab, 'labeled')

            feat_Maps_unlab, _ = self.uscl(feats_unlab, updated_Qs_lab, 'unlabeled')
            
            return output_lab, output_unlab, feat_Maps_lab, feat_Maps_unlab, feat_Maps_consisunlab
            
class InherentConsistent(nn.Module):
    def __init__(
        self,
        in_chans: Sequence[int], # different in_chans for different scales of feature maps
        depths: Sequence[int],
        patch_size: Sequence[int],
        input_resolution: Sequence[int], 
        num_classes: int,
        num_heads: Sequence[int],
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        spatial_dims: int = 2,
        drop_path_rate: float = 0.1

    ) -> None:
        super().__init__()  
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.depth = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # Tokenized Projection --> Normalized Layers --> Class Decoders --> Conv to fuse multi-head features
        self.proj_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.class_decoders = nn.ModuleList()
        self.attn_convs0 = nn.ModuleList()
        self.attn_convs1 = nn.ModuleList()
        self.query_convs = nn.ModuleList()
        
        for i_layer in range(len(depths)):
            self.proj_layers.append(Conv[Conv.CONV, spatial_dims](in_channels=in_chans[i_layer], out_channels=in_chans[i_layer], kernel_size=(1,1), stride=(1,1)))
            self.norm_layers.append(norm_layer(in_chans[i_layer]))
            self.class_decoders.append(Class_Decoder(dim=in_chans[i_layer], input_resolution=ensure_tuple_rep(input_resolution[i_layer], 2),
                                 num_heads=num_heads[i_layer],
                                 mlp_ratio=4.,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 drop_path=dpr[1],
                                 norm_layer=norm_layer))
            self.attn_convs0.append(SeparableConv2d(num_heads[i_layer], num_heads[i_layer], (3,3), norm_layer=nn.BatchNorm2d,relu_first=False))                                
            self.attn_convs1.append(nn.Conv2d(in_channels=num_heads[i_layer], out_channels=1, kernel_size=(1,1), stride=(1,1)))
            self.query_convs.append(nn.Conv1d(in_channels=in_chans[i_layer], out_channels=in_chans[i_layer]//2, kernel_size=1, stride=1, padding=0))
        # Global Guidance Initialized -->
        self.guided_Q = nn.Parameter(torch.zeros(1, num_classes, in_chans[0]))

    def forward(self, feats, guided_Q=None, modal='labeled'):
        feat_maps = []
        updated_Qs = []
        BS = feats[0].shape[0]
        if modal == 'labeled':
            next_guided_Q = self.guided_Q.expand(BS, -1, -1)
        # Tokenized --> Normalized --> Class Decoders --> Conv to fuse multi-head features
            for i_layer in range(len(self.depth)):
                tok_feats = self.norm_layers[i_layer](self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
                updated_guided_Q, attn_map = self.class_decoders[i_layer](next_guided_Q, tok_feats)
                bs, num_classes, num_heads, N_patch = attn_map.size()
                h, w = int(np.sqrt(N_patch)), int(np.sqrt(N_patch))
                attn_map = attn_map.contiguous().view(bs, num_classes, num_heads, h, w)
                attn_map = attn_map.reshape(bs*num_classes, num_heads, h, w)
                attn_map = self.attn_convs0[i_layer](attn_map)
                feat_map = self.attn_convs1[i_layer](attn_map).squeeze(1).reshape(bs, num_classes, h, w)
                next_guided_Q = self.query_convs[i_layer](updated_guided_Q.permute(0, 2, 1)).squeeze(1)
                next_guided_Q = next_guided_Q.permute(0, 2, 1)

                feat_maps.append(feat_map)
                updated_Qs.append(updated_guided_Q.mean(dim=0, keepdim=True))
            return feat_maps, updated_Qs

        elif modal == 'unlabeled':
            for i_layer in range(len(self.depth)):
                tok_feats = self.norm_layers[i_layer](self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
                updated_guided_Q, attn_map = self.class_decoders[i_layer](guided_Q[i_layer].expand(BS, -1, -1), tok_feats)
                bs, num_classes, num_heads, N_patch = attn_map.size()
                h, w = int(np.sqrt(N_patch)), int(np.sqrt(N_patch))
                attn_map = attn_map.contiguous().view(bs, num_classes, num_heads, h, w)
                attn_map = attn_map.reshape(bs*num_classes, num_heads, h, w)
                attn_map = self.attn_convs0[i_layer](attn_map)
                feat_map = self.attn_convs1[i_layer](attn_map).squeeze(1).reshape(bs, num_classes, h, w)
                next_guided_Q = self.query_convs[i_layer](updated_guided_Q.permute(0, 2, 1)).squeeze(1)
                next_guided_Q = next_guided_Q.permute(0, 2, 1)

                feat_maps.append(feat_map)
                updated_Qs.append(updated_guided_Q.mean(dim=0, keepdim=True))
            return feat_maps, updated_Qs

class Class_Decoder(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                     attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_query = norm_layer(dim)
        self.attn = Query_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(input_resolution[0]*input_resolution[1])
        self.mlp2 = MLP(in_features=input_resolution[0]*input_resolution[1], hidden_features=input_resolution[0]*input_resolution[1], act_layer=act_layer, drop=drop)
    def forward(self, query, feat):
        query, attn = self.attn(self.norm1_query(query), self.norm1(feat))
        query = query + self.drop_path(query)
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        attn = attn + self.drop_path(attn)
        attn = attn + self.drop_path(self.mlp2(self.norm3(attn)))
        return query, attn

class Query_Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.fc_q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.fc_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        B, N, C = x.shape
        num_classes = q.shape[1]
        q = self.fc_q(q).reshape(B, self.num_heads, num_classes, C // self.num_heads)
        kv = self.fc_kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_head, N, C/num_head]
        attn1 = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_head, class, N]
        attn2 = attn1.softmax(dim=-1)
        attn3 = self.attn_drop(attn2)  # [B, num_head, 12, N]
        x = (attn3 @ v).reshape(B, num_classes, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # [B, 12, 256]
        attn=attn1.permute(0, 2, 1, 3)
        return x,attn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

