# adaptation of https://github.com/czczup/ViT-Adapter

import logging
import math
import timeit
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from nets.dino_v2_with_adapter.dino_v2.layers import MemEffAttention
from nets.dino_v2_with_adapter.dino_v2.layers import NestedTensorBlock as Block
from nets.dino_v2_with_adapter.dino_v2.model.vision_transformer import (
    DinoVisionTransformer,
)
from nets.ops.modules import MSDeformAttn

from .adapter_modules import (
    InteractionBlock,
    InteractionBlockWithCls,
    SpatialPriorModule,
    deform_inputs,
)

_logger = logging.getLogger(__name__)


class DinoAdapter(DinoVisionTransformer):
    # patch_size in dino_v2: 14 instead of 16
    def __init__(self, num_heads=12, pretrain_size=518, pretrained_vit=True, patch_size=14, embed_dim=768, depth=12,
                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention), conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                 with_cffn=True, cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=False, pretrained=None,
                 use_extra_extractor=True, with_cp=False, freeze_dino=True, *args, **kwargs):

        super().__init__(img_size=pretrain_size, num_heads=num_heads, patch_size=14, embed_dim=embed_dim, depth=depth, mlp_ratio=mlp_ratio,
                         block_fn=block_fn, freeze=True, *args, **kwargs)

        self.pretrain_size = pretrain_size

        if pretrained_vit:
            if num_heads == 6:
                # print("loading dinov2 VIT-S14 checkpoint... ")
                url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
                state_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu'))
            else:
                # print("loading dinov2 VIT-B14 checkpoint... ")
                url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
                state_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu'))

            self.load_state_dict(state_dict=state_dict, strict=True)
            # print("...dinov2 checkpoint loaded!")

        if freeze_dino:
            for param in self.parameters():
                param.requires_grad = False

        # self.num_classes = 80
        self.mask_token = None
        self.num_block = len(self.blocks)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        # embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)

        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                    init_values=init_values, drop_path=self.drop_path_rate,
                                    norm_layer=self.norm_layer, with_cffn=with_cffn,
                                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                    extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                                    with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02) uses timm --> not wanted
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size // 14, self.pretrain_size // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        W_vit = w // self.patch_size
        H_vit = h // self.patch_size
        W_adapt = w // 16
        H_adapt = h // 16

        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        x = x + pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, 0]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H_adapt, W_adapt)
            outs.append(x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]
        # 448 x 798
        c2 = c2.transpose(1, 2).view(bs, dim, H_adapt * 2, W_adapt * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_adapt, W_adapt).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_adapt // 2, W_adapt // 2).contiguous()

        c1 = self.up(c2) + c1

        x_out = x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous()

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            # x1 = F.interpolate(x1, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            # x2 = F.interpolate(x2, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            # x3 = F.interpolate(x3, size=c3.shape[-2:], mode='bilinear', align_corners=False)
            # x4 = F.interpolate(x4, size=c4.shape[-2:], mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4], x_out
