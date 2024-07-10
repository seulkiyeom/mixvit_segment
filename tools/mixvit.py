# Copyright (c) Nota AI GmbH. All rights reserved.
import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList, Sequential
from torch.nn import functional as F
from mmpretrain.models.utils import SELayer

from mmseg.registry import MODELS

class ConvBN2d(Sequential):
    """An implementation of Conv2d + BatchNorm2d with support of fusion.

    Modified from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolution kernel.
            Default: 1.
        stride (int): The stride of the convolution.
            Default: 1.
        padding (int): The padding of the convolution.
            Default: 0.
        dilation (int): The dilation of the convolution.
            Default: 1.
        groups (int): The number of groups in the convolution.
            Default: 1.
        bn_weight_init (float): The initial value of the weight of
            the nn.BatchNorm2d layer. Default: 1.0.
        init_cfg (dict): The initialization config of the module.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False))
        bn = nn.BatchNorm2d(num_features=out_channels)
        # bn initialization
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        conv2d, bn2d = self._modules.values()
        w = bn2d.weight / (bn2d.running_var + bn2d.eps)**0.5
        w = conv2d.weight * w[:, None, None, None]
        b = bn2d.bias - bn2d.running_mean * bn2d.weight / \
            (bn2d.running_var + bn2d.eps)**0.5

        m = nn.Conv2d(
            in_channels=w.size(1) * self.conv.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class PatchEmbedding(Sequential):
    def __init__(self, in_chans, dim):
        super().__init__()
        self.add_module('conv1', ConvBN2d(in_chans, dim // 8, 3, 2, 1))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', ConvBN2d(dim // 8, dim // 4, 3, 2, 1))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', ConvBN2d(dim // 4, dim // 2, 3, 2, 1))
        self.add_module('relu3', nn.ReLU())
        self.add_module('conv4', ConvBN2d(dim // 2, dim, 3, 2, 1))
        self.patch_size = 16
    
class PatchMerging(nn.Module):
    """Patch Merging for EfficientViT"""
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = ConvBN2d(dim, hid_dim, kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = ConvBN2d(hid_dim, 
                              hid_dim, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1, 
                              groups=hid_dim)
        self.se = SELayer(hid_dim,  ratio=4)
        self.conv3 = ConvBN2d(hid_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x
    
class ResidualDrop(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
class ConvMlp(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = ConvBN2d(ed, h)
        self.act = nn.ReLU()
        self.pw2 = ConvBN2d(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
    
class CascadedGroupAttention(BaseModule):
    """ Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            kernels=(1, 3, 5, 7),
            init_cfg=None
        ):
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        self.qkvs = ConvBN2d(dim, dim // 2)
        # self.dws = ConvBN2d(dim // 4, dim // 4, kernels[0], 1, kernels[0]//2, groups=dim // 4)
        self.dws = ConvBN2d(dim // 4, dim // 4, max(kernels), 1, max(kernels)//2, groups=dim // 4)

        # ks = [1, 3, 5, 5] #Depthwise 방식? 좋지 못한듯 (reuse_depthwise.txt 참고)
        self.split_out_channels = self.split_layer(dim) #Depth-wise Conv 적용 (without Pointwise Conv)
        mix = []
        for idx in range(num_heads):
            kernel_size = kernels[idx]
            pad = (kernel_size - 1) // 2
            assert self.split_out_channels[idx] == self.d
            #Depthwise Convolution: Spatial feature learning
            mix.append(torch.nn.Conv2d(self.split_out_channels[idx], self.d, kernel_size=kernel_size, padding=pad, groups=self.d))
            # mix.append(ConvBN2d(self.split_out_channels[idx], self.d, kernel_size=kernel_size, padding=pad, groups=self.d))

        self.mix = ModuleList(mix)

        self.proj = Sequential(
            nn.ReLU(),
            ConvBN2d(self.d * num_heads, 
                     dim,
                     bn_weight_init=0))

        points = list(
            itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer(
            'attention_bias_idxs',
            torch.LongTensor(idxs).view(N, N),
            persistent=False)
        self.attention_bias_cache = {}

        # valuescale = False
        # gamma_init_values=1e-5
        # self.valuescale = valuescale
        # if valuescale:
        #     self.gamma = torch.nn.Parameter(gamma_init_values * torch.ones(self.num_heads), requires_grad=True)
        
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def split_layer(self, total_channels):
        import numpy as np
        split = [int(np.ceil(total_channels / self.num_heads)) for _ in range(self.num_heads)]
        split[self.num_heads - 1] += total_channels - sum(split)
        return split


    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        B, C, H, W = x.shape
        attn_bias = self.get_attention_biases(x.device)

        feat = self.qkvs(x)
        q, k = feat.view(B, -1, H, W).split([C // 4, C // 4], dim=1) # B, C, H, W
        q = self.dws(q)
        q, k = q.flatten(2), k.flatten(2) # B, C, N

        q = q * self.scale
        attn = q.transpose(-2, -1) @ k
        attn = attn + attn_bias[0]
        attn = attn.softmax(dim=-1)

        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []

        for head_idx, vs in enumerate(self.mix):
            # feat = feat + feats_in[i] #cascading 방식 (with residual connection)
            feat = feats_in[head_idx] #with residual connection
                
            if isinstance(vs, nn.Sequential): #if bottleneck is included 
                v = vs(feat)
            else:  #normal process (fully-connected or pruned model)
                v = vs(feat) if self.d == vs.in_channels else vs(feat[:, vs.in_index])

            v = v.flatten(2)

            # if self.valuescale:
            #     feat = self.gamma[i] * (v @ attn.transpose(-2, -1)).view(B, -1, H, W) # BCHW
            # else:
                # feat = (v @ attn.transpose(-2, -1)).view(B, self.value_dim, H, W) # BCHW
            feat = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) # BCHW

            feats_out.append(feat)

        x = self.proj(torch.cat(feats_out, 1))
        # x = 0.5 * x + 0.5 * x.mean(dim=[2,3], keepdim=True) #Uniform attention
        return x
        
class LocalWindowAttention(BaseModule):
    """ Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=(5, 5, 5, 5),
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        self.attn = CascadedGroupAttention(
            dim, key_dim, num_heads,
            attn_ratio=attn_ratio,
            resolution=window_resolution,
            kernels=kernels,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1) #B, H, W, C
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W % self.window_resolution) % self.window_resolution
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3)
            x = x.reshape(B * nH * nW, self.window_resolution, self.window_resolution, C).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution, C)
            x = x.transpose(2, 3).reshape(B, pH, pW, C)
            x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x

class EfficientVitBlock(torch.nn.Module):
    """ A basic EfficientVit building block.

    Args:
        dim (int): Number of input channels.
        key_dim (int): Dimension for query and key in the token mixer.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=[5, 5, 5, 5],
        ):
        super().__init__()

        self.dw0 = ResidualDrop(ConvBN2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bn_weight_init=0.))
        self.ffn0 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

        self.mixer = ResidualDrop(
            LocalWindowAttention(
                dim, key_dim, num_heads,
                attn_ratio=attn_ratio,
                resolution=resolution,
                window_resolution=window_resolution,
                kernels=kernels,
            )
        )

        self.dw1 = ResidualDrop(ConvBN2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bn_weight_init=0.))
        self.ffn1 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))

class EfficientVitStage(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            downsample=('', 1),
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            window_resolution=7,
            kernels=[5, 5, 5, 5],
            depth=1,
    ):
        super().__init__()
        if downsample[0] == 'subsample':
            self.resolution = (resolution - 1) // downsample[1] + 1
            down_blocks = []
            down_blocks.append((
                'res1',
                Sequential(
                    ResidualDrop(ConvBN2d(in_dim, in_dim, 3, 1, 1, groups=in_dim)),
                    ResidualDrop(ConvMlp(in_dim, int(in_dim * 2))),
                )
            ))
            down_blocks.append(('patchmerge', PatchMerging(in_dim, out_dim)))
            down_blocks.append((
                'res2',
                Sequential(
                    ResidualDrop(ConvBN2d(out_dim, out_dim, 3, 1, 1, groups=out_dim)),
                    ResidualDrop(ConvMlp(out_dim, int(out_dim * 2))),
                )
            ))
            self.downsample = Sequential(OrderedDict(down_blocks))
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()
            self.resolution = resolution

        blocks = []
        for d in range(depth):
            blocks.append(EfficientVitBlock(out_dim, key_dim, num_heads, attn_ratio, self.resolution, window_resolution, kernels))
        self.blocks = Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x
    
def fuse_parameters(module):
    for child_name, child in module.named_children():
        if hasattr(child, 'fuse'):
            setattr(module, child_name, child.fuse())
        else:
            fuse_parameters(child)

@MODELS.register_module()
class MixViT(BaseModule):
    arch_settings = {
        'm0': { #ReViT_XS
            'img_size': 400,
            'embed_dim': [64, 128, 192],
            'depth': [1, 2, 2],
            'num_heads': [4, 4, 4],
            'window_size': [7, 7, 7],
            'kernels': [1, 3, 5, 7],
        },
        'm1': { #ReViT_S
            'img_size': 400,
            'embed_dim': [128, 144, 192],
            'depth': [1, 2, 2],
            'num_heads': [4, 4, 4],
            'window_size': [7, 7, 7],
            'kernels': [1, 3, 5, 7]
        },
        'm3': { #ReViT_M
            'img_size': 400,
            'embed_dim': [128, 240, 320],
            'depth': [1, 2, 2],
            'num_heads': [4, 3, 4],
            'window_size': [7, 7, 7],
            'kernels': [1, 3, 5, 7]
        },
        'm5': { #ReViT_L
            'img_size': 400,
            'embed_dim': [192, 288, 384],
            'depth': [1, 2, 3],
            'num_heads': [3, 3, 4],
            'window_size': [7, 7, 7],
            'kernels': [1, 3, 5, 7]
        },
    }

    def __init__(
            self,
            arch = 'm0',
            in_chans=3,
            key_dim=(16, 16, 16, 16),
            down_ops=(('', 1), ('subsample', 2), ('subsample', 2)),
            global_pool='avg',
            frozen_stages=-1,
            drop_rate=0.,
            deploy=False,
            pretrained=None,
            init_cfg=None
    ):
        super(MixViT, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            default_keys = set(self.arch_settings['l1'].keys())
            assert set(arch.keys()) == default_keys, \
                f'The arch dict must have {default_keys}, ' \
                f'but got {list(arch.keys())}.'
            
        img_size = arch['img_size']
        embed_dim = arch['embed_dim']
        depth = arch['depth']
        num_heads = arch['num_heads']
        window_size = arch['window_size']
        kernels = arch['kernels']

        self.grad_checkpointing = False
        self.drop_rate = drop_rate

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_chans, embed_dim[0])
        stride = self.patch_embed.patch_size
        resolution = img_size // self.patch_embed.patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]

        # Build EfficientVit blocks
        self.feature_info = []
        stages = []
        pre_ed = embed_dim[0]
        for i, (ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            stage = EfficientVitStage(
                in_dim=pre_ed,
                out_dim=ed,
                key_dim=kd,
                downsample=do,
                num_heads=nh,
                attn_ratio=ar,
                resolution=resolution,
                window_resolution=wd,
                kernels=kernels,
                depth=dpth,
            )
            pre_ed = ed
            if do[0] == 'subsample' and i != 0:
                stride *= do[1]
            resolution = stage.resolution
            stages.append(stage)
            self.feature_info += [dict(num_chs=ed, reduction=stride, module=f'stages.{i}')]
        self.stages = Sequential(*stages)

        # if global_pool == 'avg':
        #     self.global_pool = nn.AdaptiveAvgPool2d(1)
        #     self.flatten = nn.Flatten(1)
        # else:
        #     assert num_classes == 0
        #     self.global_pool = nn.Identity()
        self.num_features = embed_dim[-1]

        self.deploy = False
        if deploy:
            self.switch_to_deploy()

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    # @torch.jit.ignore
    # def get_classifier(self):
    #     return self.head.linear

    def switch_to_deploy(self):
        if self.deploy:
            return
        fuse_parameters(self)
        self.deploy = True

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for name, param in stage.named_parameters(): #freeze all parameters excluding value module
                if not any(sub in name for sub in ['attn.mix', 'attn.proj']):
                    param.requires_grad = False

    def forward_features(self, x):
        x = self.patch_embed(x)
        out = []
        for block in self.stages:
            x = block(x)
            out.append(x)
        return tuple(out)

    # def forward_head(self, x, pre_logits: bool = False):
    #     return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x

    def train(self, mode=True):
        super(MixViT, self).train(mode)
        self._freeze_stages()