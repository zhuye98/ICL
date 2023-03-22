# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math
from typing import Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3_CT
from monai.networks.layers import DropPath, Conv
from monai.utils import ensure_tuple_rep
from collections import OrderedDict
import numpy as np
import pdb


class unet_3D_icl(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_icl, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

        icl_in_chans = (filters[4], filters[3], filters[2])
        icl_in_resolutions = [6, 12, 24]

        self.sspa = InherentConsistent(
            in_chans=icl_in_chans, # different in_chans for different scales of feature maps
            depths=(2, 2, 2),
            patch_size =ensure_tuple_rep(2, 3), 
            input_resolution=icl_in_resolutions,
            num_classes=n_classes,
            num_heads=(16, 8, 4),
            norm_layer=nn.LayerNorm,
        )
        self.uscl = InherentConsistent(
            in_chans=icl_in_chans, # different in_chans for different scales of feature maps
            depths=(2, 2, 2),
            patch_size =ensure_tuple_rep(2, 3), 
            input_resolution=icl_in_resolutions,
            num_classes=n_classes,
            num_heads=(16, 8, 4),
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x_lab, x_unlab=None, inference=None):
        conv1 = self.conv1(x_lab)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center_lab = self.dropout1(center)

        up4_lab = self.up_concat4(conv4, center_lab)
        up3_lab = self.up_concat3(conv3, up4_lab)
        up2_lab = self.up_concat2(conv2, up3_lab)
        up1_lab = self.up_concat1(conv1, up2_lab)
        up1_lab = self.dropout2(up1_lab)
        final_lab = self.final(up1_lab)

        if inference:
            return final_lab
        else:
            conv1 = self.conv1(x_unlab)
            maxpool1 = self.maxpool1(conv1)
            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)
            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)
            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            center = self.center(maxpool4)
            center_unlab = self.dropout1(center)

            up4_unlab = self.up_concat4(conv4, center_unlab)
            up3_unlab = self.up_concat3(conv3, up4_unlab)
            up2_unlab = self.up_concat2(conv2, up3_unlab)
            up1_unlab = self.up_concat1(conv1, up2_unlab)
            up1_unlab = self.dropout2(up1_unlab)
            final_unlab = self.final(up1_unlab)

            feats_lab = [center_lab, up4_lab, up3_lab]
            feats_unlab = [center_unlab, up4_unlab, up3_unlab]

            feat_Maps_lab, updated_Qs_lab = self.sspa(feats_lab, 'labeled')
            feat_Maps_consis, _ = self.sspa(feats_unlab, 'labeled')
            
            feat_Maps_unlab, _ = self.uscl(feats_unlab, updated_Qs_lab, 'unlabeled')
            return final_lab, final_unlab, feat_Maps_lab, feat_Maps_unlab, feat_Maps_consis

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
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
        spatial_dims: int = 3,
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
            self.proj_layers.append(Conv[Conv.CONV, spatial_dims](in_channels=in_chans[i_layer], out_channels=in_chans[i_layer], kernel_size=(1,1,1), stride=(1,1,1)))
            self.norm_layers.append(norm_layer(in_chans[i_layer]))
            self.class_decoders.append(Class_Decoder(dim=in_chans[i_layer], input_resolution=ensure_tuple_rep(input_resolution[i_layer], 3),
                                 num_heads=num_heads[i_layer],
                                 mlp_ratio=4.,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 drop_path=dpr[1],
                                 norm_layer=norm_layer))
            self.attn_convs0.append(SeparableConv3d(num_heads[i_layer], num_heads[i_layer], (3,3,3), norm_layer=nn.BatchNorm3d,relu_first=False))                                
            self.attn_convs1.append(nn.Conv3d(in_channels=num_heads[i_layer], out_channels=1, kernel_size=(1,1,1), stride=(1,1,1)))
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
                d, h, w = int(np.cbrt(N_patch)), int(np.cbrt(N_patch)), int(np.cbrt(N_patch))
                attn_map = attn_map.contiguous().view(bs, num_classes, num_heads, d, h, w)
                attn_map = attn_map.reshape(bs*num_classes, num_heads, d, h, w)
                attn_map = self.attn_convs0[i_layer](attn_map)
                feat_map = self.attn_convs1[i_layer](attn_map).squeeze(1).reshape(bs, num_classes, d, h, w)
                next_guided_Q = self.query_convs[i_layer](updated_guided_Q.permute(0, 2, 1)).squeeze(1)
                next_guided_Q = next_guided_Q.permute(0, 2, 1)

                feat_maps.append(feat_map)
                updated_Qs.append(updated_guided_Q.mean(dim=0, keepdim=True))
        
        elif modal == 'unlabeled':
            for i_layer in range(len(self.depth)):
                tok_feats = self.norm_layers[i_layer](self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
                updated_guided_Q, attn_map = self.class_decoders[i_layer](guided_Q[i_layer].expand(BS, -1, -1), tok_feats)
                bs, num_classes, num_heads, N_patch = attn_map.size()
                d, h, w = int(np.cbrt(N_patch)), int(np.cbrt(N_patch)), int(np.cbrt(N_patch))
                attn_map = attn_map.contiguous().view(bs, num_classes, num_heads, d, h, w)
                attn_map = attn_map.reshape(bs*num_classes, num_heads, d, h, w)
                attn_map = self.attn_convs0[i_layer](attn_map)
                feat_map = self.attn_convs1[i_layer](attn_map).squeeze(1).reshape(bs, num_classes, d, h, w)
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

        self.norm3 = norm_layer(input_resolution[0]*input_resolution[1]*input_resolution[2])
        self.mlp2 = MLP(in_features=input_resolution[0]*input_resolution[1]*input_resolution[2], hidden_features=input_resolution[0]*input_resolution[1]*input_resolution[2], act_layer=act_layer, drop=drop)
    def forward(self, query, feat):
        # query:[B,14,384] feat:[B,196,384] attn: [B,14,12(head),196]

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
        # q:[B,Class,C] x:[B,N,C]
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

class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,1,1), relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm3d):
        super().__init__()
        depthwise = nn.Conv3d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv3d(inplanes, planes, (1,1,1), bias=bias)
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