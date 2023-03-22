# coding=utf-8
# This file borrowed from Swin-UNet: https://github.com/HuCaoFighting/Swin-Unet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from turtle import back
import timm
import pdb

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from typing import Sequence, Type, Any, Tuple
from collections.abc import Iterable
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from networks.swinunet_icl import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
        icl_in_chans = (384, 192, 96)
        icl_in_resolutions = (14, 28, 56)
        icl_num_heads = (24, 12, 6)
        depths = (2, 2, 2)
        
        self.sspa = InherentConsistent(
            in_chans=icl_in_chans, # different in_chans for different scales of feature maps
            depths=depths,
            patch_size=config.MODEL.SWIN.PATCH_SIZE, 
            input_resolution=icl_in_resolutions,
            num_classes=self.num_classes,
            num_heads=icl_num_heads,
            norm_layer=nn.LayerNorm,
        )
        self.uscl = InherentConsistent(
            in_chans=icl_in_chans, # different in_chans for different scales of feature maps
            depths=depths,
            patch_size=config.MODEL.SWIN.PATCH_SIZE, 
            input_resolution=icl_in_resolutions,
            num_classes=self.num_classes,
            num_heads=icl_num_heads,
            norm_layer=nn.LayerNorm,
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_lab, x_unlab=None, inference=False):
        if inference:
            if x_lab.size()[1] == 1:
                x_lab = x_lab.repeat(1,3,1,1)
            output_lab = self.swin_unet(x_lab, inference=inference)
            return output_lab

        else:
            if x_lab.size()[1] == 1 and x_unlab.size()[1] == 1:
                x_lab = x_lab.repeat(1,3,1,1)
                x_unlab = x_unlab.repeat(1,3,1,1)
                
                output_lab, output_unlab, feats_lab, feats_unlab = self.swin_unet(x_lab, x_unlab)
                
                feat_Maps_lab, updated_Qs_lab = self.sspa(feats_lab, 'labeled')
                feat_Maps_consisunlab, _ = self.sspa(feats_unlab, 'labeled')
                feat_Maps_unlab, _ = self.uscl(feats_unlab, updated_Qs_lab, 'unlabeled')
                
                return output_lab, output_unlab, feat_Maps_lab, feat_Maps_unlab, feat_Maps_consisunlab
        

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))
 
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
            self.proj_layers.append(nn.Conv2d(in_channels=in_chans[i_layer], out_channels=in_chans[i_layer], kernel_size=(1,1), stride=(1,1)))
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
        # No need for Tokenization,  --> Class Decoders --> Conv to fuse multi-head features
            for i_layer in range(len(self.depth)):
                #tok_feats = self.norm_layers[i_layer](self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
                updated_guided_Q, attn_map = self.class_decoders[i_layer](next_guided_Q, feats[i_layer])
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
                #tok_feats = self.norm_layers[i_layer](self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
                updated_guided_Q, attn_map = self.class_decoders[i_layer](guided_Q[i_layer].expand(BS, -1, -1), feats[i_layer])
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