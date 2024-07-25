import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.append("..")

from torchvision.models.resnet import resnet18

import utils.basic
import utils.geom
import utils.misc
import utils.vox
from nets.dino_v2_with_adapter.dino_v2_adapter.dinov2_adapter import (
    DinoAdapter,
)

EPS = 1e-4


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        # x_to_upsample=x2 , x=x1
        # x2 is bilinearly upsampled by factor 2 to match the output size of x1
        x_to_upsample = self.upsample(x_to_upsample)
        # concatenation -> x1 channels(512) + x2 channels (1024) = 1536 channels
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        # return the output of a self defined convolution layer
        return self.conv(x_to_upsample)  # in_channels=1536, out_channels=512


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip


class Decoder(nn.Module):
    def __init__(self, in_channels, task, n_classes):
        super().__init__()
        # to avoid torchvision deprecation warning for the parameter "pretrained=False"
        # backbone = resnet18(pretrained=False, zero_init_residual=True)
        backbone = resnet18(weights=None, zero_init_residual=True)
        # changes in_channels from 3(resnet-18 to 128
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)  # 128 -> 64; HW -> HW/2
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        # add bev map classes
        self.n_classes = n_classes
        self.task = task

        # maxpool from original resnet-18 is omitted

        self.layer1 = backbone.layer1  # 64 -> 64;     HW/2 -> HW/2 (unchanged)
        self.layer2 = backbone.layer2  # 64 -> 128;    HW/2 -> HW/4
        self.layer3 = backbone.layer3  # 128 -> 256;   HW/4 -> HW/8

        # layer 4 and final pooling + fc layer are omitted

        shared_out_channels = in_channels  # 128
        # definition of additive skip connections
        # - it first upsamples the maps by factor 2 in H and W
        # - then 1x1 convolution -> only reduce number of channels
        # - instance norm along channel dim
        # - in forward pass: add upsampled data to skipped data
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)  # HW/8 -> HW/4
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)  # HW/4 -> HW/2
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)  # HW/2 -> HW

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1),
        )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W) -> (200,200)
        skip_x = {'1': x}  # first skip connection before first layer
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x  # skip connection before layer 2
        x = self.layer2(x)
        skip_x['3'] = x  # skip connection before layer 3

        # (H/8, W/8)
        x = self.layer3(x)  # output after last decoder layer

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])  # upsamples x to match dims of layer2 output and adds them (+conv)

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])  # upsamples x to match dims of layer1 output and adds them (+conv)

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])  # upsamples x to match dims of first layer output and adds them (+conv)

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        # apply task specific head
        # run model output through respective heads
        out_dict = {}
        segmentation_output = self.segmentation_head(x)  # (B,X,200,200)
        if self.task == 'object_decoder':
            out_dict = {
                'obj_segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),  # (B,1,200,200)
            }
        elif self.task == 'map_decoder':
            out_dict = {
                'bev_map_segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
                # (B,7,200,200)
            }
        elif self.task == 'shared_decoder':
            out_dict = {
                # (B,7,200,200)
                'bev_map_segmentation': segmentation_output[:, :-1],
                # (B,1,200,200)
                'obj_segmentation': segmentation_output[:, -1:],
            }
        return out_dict


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C  # C = 128 (latent_dim)
        # to avoid torchvision deprecation warning for the parameter "pretrained"
        # resnet = torchvision.models.resnet101(pretrained=True)
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        # get all layers except the last 4
        # -> we don't use the average pooling laver and all three blocks of layer 4 from the original ResNet
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])

        # explicitly create a layer of the type block 3 from resnet-101
        # layer 3_x:
        #   conv1 1x1, 256
        #   conv2 3x3, 256
        #   conv3 1x1, 1024
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)  # passes input in net and runs it through all layers -> x1=output of the model
        x2 = self.layer3(x1)  # define x2 to be the output of layer 3
        x = self.upsampling_layer(x2, x1)  # input: x_to_upsample=x2 (1024 channels), x=x1 (512 channels)
        # in_channels=1536 (output of layer3 = 1024 + x1 = 512), out_channels=512
        x = self.depth_layer(x)  # 1x1 convolution from 512 channels to 128 (no padding)

        return x


class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class DinoMulti2SingleScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.single_scale_compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),  # ReLU throws backprob. error maybe due to "dying ReLU" problem !!!
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x_4, x_8, x_16, x_32):
        x_4_8 = torch.nn.functional.interpolate(input=x_4, scale_factor=0.5, mode='bilinear')
        x_16_8 = self.up_2(x_16)
        x_32_8 = self.up_4(x_32)
        x = torch.cat([x_4_8, x_8, x_16_8, x_32_8], dim=1)
        x = self.single_scale_compress(x)
        # print("SHAPE OF FUSED MULTISCALE FEATS: " + str(x.shape))
        return x


class SegnetWithMap(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None,
                 use_radar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101",
                 use_shallow_metadata=False,
                 train_task='both',
                 freeze_dino=True,
                 is_master=False
                 ):
        super(SegnetWithMap, self).__init__()
        assert (encoder_type in ["res101", "res50", "dino_v2"])

        self.Z, self.Y, self.X = Z, Y, X  # Z=200, Y=8, X=200
        self.use_radar = use_radar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        # custom args
        self.use_shallow_metadata = use_shallow_metadata
        self.train_task = train_task
        self.freeze_dino = freeze_dino
        self.is_master = is_master

        # mean and std for every color channel
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)  # using this backbone (feat2d_dim = 128)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        else:  # if encoder_type == "dino_v2":
            self.encoder = DinoAdapter(add_vit_feature=False, pretrain_size=518, pretrained_vit=True,
                                       freeze_dino=freeze_dino)

            self.img_feats_compr_4 = nn.Sequential(
                nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                          kernel_size=1, stride=1, bias=True),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
            self.img_feats_compr_8 = nn.Sequential(
                nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                          kernel_size=1, stride=1, bias=True),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
            self.img_feats_compr_16 = nn.Sequential(
                nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                          kernel_size=1, stride=1, bias=True),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
            self.img_feats_compr_32 = nn.Sequential(
                nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=latent_dim,
                          kernel_size=1, stride=1, bias=True),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )

            self.dino_ms_fuse = DinoMulti2SingleScale(in_channels=4 * latent_dim, out_channels=latent_dim)

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    # Y = 8 --> vertical dimension extends the channel dimension
                    nn.Conv2d(feat2d_dim * Y + 16 * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            elif self.use_shallow_metadata:
                self.bev_compressor = nn.Sequential(
                    # Y = 8 --> vertical dimension extends the channel dimension
                    nn.Conv2d(feat2d_dim * Y + 4 * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * Y + 1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        """
            class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
            class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
            class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
            class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
            class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
            class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
            class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n
            other -> considered background
        """

        if self.train_task == "object":
            self.object_decoder = Decoder(in_channels=self.latent_dim, task='object_decoder', n_classes=1)
            # Weights
            self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.train_task == "map":
            self.map_decoder = Decoder(in_channels=self.latent_dim, task='map_decoder', n_classes=7)

            self.fc_map_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.train_task == "both":
            self.shared_decoder = Decoder(in_channels=self.latent_dim, task='shared_decoder', n_classes=8)
            self.fc_map_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            # transforms mem coordinates into ref coordinates
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
        """
        B, S, C, H, W = rgb_camXs.shape
        assert (C == 3)
        # reshape tensors
        # __p, __u: x is a tensor, B is the batch size
        # pack_seqdim: reshaping: (B,S,C,H,W) -> ([B*S],C,H,W)
        __p = lambda x: utils.basic.pack_seqdim(x, B)  # combines batch and number of cam dims
        # unpack_seqdim: reshaping: ([B*S],C,H,W) -> (B,S,C,H,W)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)  # splits [BS] back into [B,S]

        # __p -> "pack" input
        rgb_camXs_ = __p(rgb_camXs)  # (B,S,C,H,W)   ->  ([B*S],C,H,W)
        pix_T_cams_ = __p(pix_T_cams)  # (B,S,4,4)     ->  ([B*S],4,4)
        cam0_T_camXs_ = __p(cam0_T_camXs)  # (B,S,4,4)     ->  ([B*S],4,4)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)  # inverse of transformation matrix

        # rgb encoder
        device = rgb_camXs_.device
        # input normalization: add 0.5 and subtract the color-channel-specific mean
        # divide through the color-specific std
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)

        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            # decide which images in one batch should be flipped
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            # -1: flip on last dim -> W -> flip vertically
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])

        # put randomly flipped input data into encoder
        # image features as output of modified encoder -> 128 x H/8 x W/8
        if self.encoder_type == 'dino_v2':
            img_encoder_feats, dino_out = self.encoder(rgb_camXs_)
            # compress dino feats down to 128 channels
            feats_4_ = self.img_feats_compr_4(img_encoder_feats[0])
            feats_8_ = self.img_feats_compr_8(img_encoder_feats[1])
            feats_16_ = self.img_feats_compr_16(img_encoder_feats[2])
            feats_32_ = self.img_feats_compr_32(img_encoder_feats[3])

            # combine all feature maps into one...
            feat_camXs_ = self.dino_ms_fuse(x_4=feats_4_, x_8=feats_8_, x_16=feats_16_, x_32=feats_32_)

        else:
            feat_camXs_ = self.encoder(rgb_camXs_)

        if self.rand_flip:
            # "unflip" the image feature maps based on the same random order of the image flipping
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape  # C=128, Hf=H/8, Wf=W/8

        sy = Hf / float(H)  # sy = 1/8
        sx = Wf / float(W)  # sx = 1/8
        Z, Y, X = self.Z, self.Y, self.X  # 200, 8, 200

        # unproject image feature to 3d grid
        # featuremaps from all cameras must be "unprojected" into a 3D space around the ego car to later build the
        # BEV feature space
        # scale_intrinsics: pix_T_cams_ -> Camera Matrix: focal point and centerpoint are scaled by sx,sy respectively
        # first extract focal point and centerpoint from pix_T_cams -> then scale -> then merge into new matrix
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

        if self.xyz_camA is not None:
            # 3d mem in view of reference cam??? (in meters)
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B * S, 1, 1)
        else:
            xyz_camA = None

        # unproject_image_to_mem: transforms image features from all cams into shared 3D feature space
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)

        # unpack features from ([B*S], C, Z, Y, X) -> (B, S, C, Z, Y, X)
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X

        # mask is 1 if abs value is != 0 else zero
        mask_mems = (torch.abs(feat_mems) > 0).float()
        # S = 0 since in 3D feature space we don't need the number of cams dim -> reduce this dim
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        if self.use_radar:
            assert (rad_occ_mem0 is not None)
            if not self.use_metaradar and not self.use_shallow_metadata:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0, 1)  # squish the vertical dim
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                if self.use_shallow_metadata:
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 4 * Y, Z, X)
                else:
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y, Z, X)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
        else:  # rgb only
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        seg_e = {}

        if self.train_task == "object":
            out_dict_objects = self.object_decoder(feat_bev, (
                self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
            # object estimation data
            obj_seg_e = out_dict_objects['obj_segmentation']
            seg_e = obj_seg_e

        if self.train_task == "map":
            out_dict_map = self.map_decoder(feat_bev, (
                self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
            # map estimation data
            bev_map_seg_e = out_dict_map['bev_map_segmentation']
            seg_e = bev_map_seg_e

        if self.train_task == "both":
            out_dict_shared = self.shared_decoder(feat_bev, (
                self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
            # map estimation data
            bev_map_seg_e = out_dict_shared['bev_map_segmentation']
            obj_seg_e = out_dict_shared['obj_segmentation']
            seg_e = torch.cat([bev_map_seg_e, obj_seg_e], dim=1)  # [b, 8, 200, 200]

        return seg_e
