import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet18

import utils.basic
import utils.geom
import utils.misc
import utils.vox
from nets.dino_v2_with_adapter.dino_v2_adapter.dinov2_adapter import (
    DinoAdapter,
)
from nets.voxelnet import VoxelNet

sys.path.append("..")
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


class DownsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.5):
        super().__init__()

        self.scale_factor = scale_factor

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_downsample, x):
        x_down = torch.nn.functional.interpolate(input=x_to_downsample, scale_factor=self.scale_factor, mode='bilinear')
        # concatenation -> x1 channels(512) + x2 channels (1024) = 1536 channels
        x = torch.cat([x, x_down], dim=1)
        # return the output of a self defined convolution layer
        return self.conv(x)  # in_channels=256, out_channels=128


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


class FeatureEncoderDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # to avoid torchvision deprecation warning for the parameter "pretrained=False"
        # backbone = resnet18(pretrained=False, zero_init_residual=True)
        backbone = resnet18(weights=None, zero_init_residual=True)
        # changes in_channels from 3(resnet-18 to 128
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # kernel_size=7, stride=2, padding=3 -> kernel_size=5, stride=1, padding=2 # 128 -> 64; HW -> HW
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        # maxpool from original resnet-18 is omitted

        self.layer1 = backbone.layer1  # 64 -> 64;     HW/2 -> HW/2 (unchanged)
        self.layer2 = backbone.layer2  # 64 -> 128;    HW/2 -> HW/4
        self.layer3 = backbone.layer3  # 128 -> 256;   HW/4 -> HW/8

        # layer 4 and final pooling + fc layer are omitted

        shared_out_channels = in_channels  # 640
        # definition of additive skip connections
        # - it first upsamples the maps by factor 2 in H and W
        # - then 1x1 convolution -> only reduce number of channels
        # - instance norm along channel dim
        # - in forward pass: add upsampled data to skipped data
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)  # HW/8 -> HW/4
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)  # HW/4 -> HW/2
        self.skip_conv1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False),
                                        nn.InstanceNorm2d(128))  # HW/2 -> HW

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape  # (B,128,100,100)

        # (H, W) -> (100,100)
        skip_x = {'1': x}  # first skip connection before first layer  (B,128,100,100)
        x = self.first_conv(x)  # (B,128,100,100) -> (B,64,100,100)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)  # (B,64,100,100) -> (B,64,100,100)
        skip_x['2'] = x  # skip connection before layer 2  (B,64,100,100)
        x = self.layer2(x)  # (B,64,100,100) -> (B,128,50,50)
        skip_x['3'] = x  # skip connection before layer 3  (B,128,50,50)

        # (H/8, W/8)
        x = self.layer3(x)  # output after last decoder layer (B,128,50,50) -> (B,256,25,25)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])  # upsamples x to match dims of layer2 output and adds them (+conv)

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])  # upsamples x to match dims of layer1 output and adds them (+conv)

        # Third skip and add to (H, W)
        x = self.skip_conv1(x)  # 1x1 conv to get matching feature dim
        x = x + skip_x['1']

        return x  # (B,128,100,100)


class TaskSpecificDecoder(nn.Module):
    def __init__(self, in_channels, task, n_classes, use_feat_head=False, predict_future_flow=False,
                 use_obj_layer_only_on_map=False):
        super(TaskSpecificDecoder, self).__init__()
        self.out_channels = 128
        self.n_classes = n_classes
        self.task = task
        self.predict_future_flow = predict_future_flow
        self.use_feat_head = use_feat_head
        self.use_obj_layer_only_on_map = use_obj_layer_only_on_map

        # structure like in a ResNet18 --> one convolution block and one identity block
        self.upsample_conv_layer = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
        )

        self.upsample_skip_layer = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # here: no upsampling since transformer operates on 200x200 already
            nn.Conv2d(in_channels, 128, kernel_size=1, padding=0, bias=False),
        )

        # first conv block
        # self.conv1_1_1 = self.upsample_conv_layer
        self.first_conv_block = nn.Sequential(
            self.upsample_conv_layer,
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)   nn.BatchNorm2d(512)
            nn.ReLU(inplace=True),  # inplace=True
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)
        )
        self.skip_conv1_1 = self.upsample_skip_layer

        # second conv block
        self.second_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)   nn.BatchNorm2d(512)
            nn.ReLU(inplace=True),  # inplace=True
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)
        )
        self.skip_conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # third conv block
        self.third_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)   nn.BatchNorm2d(512)
            nn.ReLU(inplace=True),  # inplace=True
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),  # nn.InstanceNorm2d(512)
        )
        self.skip_conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        # definition of output head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_channels, self.n_classes, kernel_size=1),
        )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape  # (B,128,200,200)

        # first conv block
        # (H, W) -> (100,100)  -> (200, 200)
        skip_1_1 = self.skip_conv1_1(x)  # (B,640,100,100) -> (B,512,200,200)
        x = self.first_conv_block(x)
        x = x + skip_1_1
        x = F.relu(x, inplace=True)

        # second conv block
        skip_2_1 = self.skip_conv2_1(x)  # (B,512,200,200) -> (B,256,200,200)
        x = self.second_conv_block(x)  # (B,512,200,200) -> (B,256,200,200)
        x = x + skip_2_1
        x = F.relu(x, inplace=True)

        # third conv block
        skip_3_1 = self.skip_conv3_1(x)  # (B,256,200,200) -> (B,128,200,200)
        x = self.third_conv_block(x)  # (B,256,200,200) -> (B,128,200,200)
        x = x + skip_3_1

        # 'unflip' if flipped before
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
                'bev_map_segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),  # (B,7,200,200)
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
        # to avoid torchvision deprecation warning for the parameter "pretrained=True"
        # resnet = torchvision.models.resnet101(pretrained=True)
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        # get all layers except the last 4 -> we don't use the average pooling laver and all three blocks of layer 4 from the original ResNet
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
    def __init__(self, C, use_multi_scale_img_feats):
        super().__init__()
        self.C = C
        self.use_multi_scale_img_feats = use_multi_scale_img_feats
        resnet = torchvision.models.resnet50(pretrained=True)

        if self.use_multi_scale_img_feats:
            self.backbone = nn.Sequential(*list(resnet.children())[:-6])  # holds layer 1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3

            self.h4_2_channels = nn.Conv2d(256, self.C, kernel_size=1, padding=0)
            self.h8_2_channels = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
            self.h16_2_channels = nn.Conv2d(1024, self.C, kernel_size=1, padding=0)

        else:
            self.backbone = nn.Sequential(*list(resnet.children())[:-4])  # holds layer 1 and 2
            self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        feat_dict = {}
        if self.use_multi_scale_img_feats:
            x0 = self.backbone(x)  # (B*S, 3, H, W) --> (B*S, 64, H/4, W/4)
            x1 = self.layer1(x0)  # (B*S, 64, H/4, W/4) --> (B*S, 256, H/4, W/4)
            x2 = self.layer2(x1)  # (B*S, 256, H/4, W/4) --> (B*S, 512, H/8, W/8)
            x3 = self.layer3(x2)  # (B*S, 512, H/8, W/8) --> (B*S, 1024, H/16, W/16)

            # feature extraction with same channel depth:
            x1_ = self.h4_2_channels(x1)
            x2_ = self.h8_2_channels(x2)
            x3_ = self.h16_2_channels(x3)

            feat_dict = {
                # "feats_2": x0,
                "feats_4": x1_,
                "feats_8": x2_,
                "feats_16": x3_,
            }
        else:
            x2 = self.backbone(x)  # res:
            x3 = self.layer3(x2)  #

        x = self.upsampling_layer(x3, x2)
        x = self.depth_layer(x)
        feat_dict["output"] = x

        return feat_dict  # x


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


class SegnetSimpleLiftFuse(nn.Module):
    def __init__(self, Z_cam, Y_cam, X_cam, Z_rad, Y_rad, X_rad, vox_util=None,
                 use_radar=False,
                 use_metaradar=False,
                 use_shallow_metadata=False,
                 use_radar_encoder=False,
                 do_rgbcompress=False,
                 rand_flip=False,
                 latent_dim=128,  # 256, # 768, # 128 , 768
                 encoder_type="res101",
                 radar_encoder_type="voxel_net",
                 train_task="both",
                 use_obj_layer_only_on_map=False,
                 do_feat_enc_dec=False,  # False,
                 use_multi_scale_img_feats=False,
                 num_layers=2,
                 vis_feature_maps=False,
                 compress_adapter_output=True,
                 use_rpn_radar=False,
                 use_radar_occupancy_map=False,
                 freeze_dino=True,
                 is_master=False):
        super(SegnetSimpleLiftFuse, self).__init__()
        assert (encoder_type in ["res101", "res50", "dino_v2", "vit_s"])
        assert (radar_encoder_type in ["voxel_net", None])
        assert (train_task in ["object", "map", "both"])

        self.Z_cam, self.Y_cam, self.X_cam = Z_cam, Y_cam, X_cam  # Z=100, Y=4, X=100
        self.Z_rad, self.Y_rad, self.X_rad = Z_rad, Y_rad, X_rad  # Z=200, Y=8, X=200
        self.use_radar = use_radar
        self.use_metaradar = use_metaradar
        self.use_shallow_metadata = use_shallow_metadata
        self.use_radar_encoder = use_radar_encoder
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.radar_encoder_type = radar_encoder_type
        self.train_task = train_task
        self.use_obj_layer_only_on_map = use_obj_layer_only_on_map
        self.do_feat_enc_dec = do_feat_enc_dec
        self.use_multi_scale_img_feats = use_multi_scale_img_feats
        self.num_layers = num_layers
        self.vis_feature_maps = vis_feature_maps
        self.compress_adapter_output = compress_adapter_output

        self.use_radar_only_init = False
        self.use_rpn_radar = use_rpn_radar
        self.use_radar_occupancy_map = use_radar_occupancy_map
        self.freeze_dino = freeze_dino
        self.is_master = is_master

        if is_master:
            print("latent_dim: ", latent_dim)

        # mean and std for every color channel -> how did they obtained the values?
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Image Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)  # using this backbone (feat2d_dim = 128)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim, use_multi_scale_img_feats=use_multi_scale_img_feats)
        elif encoder_type == "dino_v2":
            self.encoder = DinoAdapter(add_vit_feature=False, pretrain_size=518, pretrained_vit=True,
                                       freeze_dino=freeze_dino)
            if self.compress_adapter_output:  # dino embed dim: 768 --> desired: 128
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

                self.dino_ms_fuse = DinoMulti2SingleScale(in_channels=4*latent_dim, out_channels=latent_dim)

        # Radar Encoder
        if self.use_radar_encoder and self.use_radar:
            if self.radar_encoder_type == "voxel_net":
                # if reduced_zx==True -> 100x100 instead of 200x200
                # if use_col=False: added RPN after CML
                self.radar_encoder = VoxelNet(use_col=self.use_rpn_radar, reduced_zx=False,
                                              output_dim=latent_dim,
                                              use_radar_occupancy_map=self.use_radar_occupancy_map)
            else:
                print("Radar encoder not found ")
        elif not self.use_radar_encoder and self.use_radar and self.is_master:
            print("#############    NO RADAR ENCODING    ##############")
        else:
            print("#############    CAM ONLY    ##############")

        # SimpleBEV based Lifting and Fusion module
        if is_master:
            print("Transformer initialized")

        self.bev_compressor = nn.Sequential(
            # Y = 8 --> vertical dimension extends the channel dimension
            nn.Conv2d(feat2d_dim * Y_cam + feat2d_dim, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
        )

        # Apply Feature Encoder - Decoder on concatenated features
        if do_feat_enc_dec:
            self.feat_enc_dec = FeatureEncoderDecoder(in_channels=self.latent_dim)

        # Decoder
        """
            class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
            class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
            class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
            class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
            class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
            class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
            class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n

            optional:
            class 7: Objects --> Vehicles as additional class for Object segmentation
            other -> considered background
        """
        if self.train_task == "object":
            self.object_decoder = TaskSpecificDecoder(in_channels=self.latent_dim,
                                                      task='object_decoder',
                                                      n_classes=1,
                                                      use_feat_head=False,
                                                      predict_future_flow=False)
            # Weights
            self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.train_task == "map":
            self.map_decoder = TaskSpecificDecoder(in_channels=self.latent_dim,
                                                   task='map_decoder',
                                                   n_classes=7,
                                                   use_feat_head=False,
                                                   predict_future_flow=False)

            self.fc_map_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        elif self.train_task == "both":
            self.shared_decoder = TaskSpecificDecoder(in_channels=self.latent_dim,
                                                      task='shared_decoder',
                                                      n_classes=8,
                                                      use_feat_head=False,
                                                      predict_future_flow=False,
                                                      use_obj_layer_only_on_map=use_obj_layer_only_on_map)

            self.fc_map_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            print("invalid task")

        if is_master:
            print("Decoder initialized")

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z_cam, Y_cam, X_cam, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z_cam, Y_cam, X_cam,
                                             assert_cube=False)  # transforms mem coordinates into ref coordinates
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

        B0, _, _, _ = rgb_camXs_.shape

        dinovoxel = None

        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            # decide which images in one batch should be flipped
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            # -1: flip on last dim -> W -> flip vertically
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])

        # put randomly flipped input data into encoder
        # image features as output of modified encoder -> 128 x H/8 x W/8
        if self.use_multi_scale_img_feats:

            if self.encoder_type == 'dino_v2':
                # we need to get the feature space down to 128 ---> do we loose too much information?
                # use convolutions for that
                img_encoder_feats, dino_out = self.encoder(rgb_camXs_)
                if self.compress_adapter_output:
                    feats_4_ = self.img_feats_compr_4(img_encoder_feats[0])
                    feats_8_ = self.img_feats_compr_8(img_encoder_feats[1])
                    feats_16_ = self.img_feats_compr_16(img_encoder_feats[2])
                    feats_32_ = self.img_feats_compr_32(img_encoder_feats[3])

                    # combine all feature maps into one...
                    feat_camXs_ = self.dino_ms_fuse(x_4=feats_4_, x_8=feats_8_, x_16=feats_16_, x_32=feats_32_)

                else:
                    feats_4 = img_encoder_feats[0]
                    feats_8 = img_encoder_feats[1]
                    feats_16 = img_encoder_feats[2]
                    feats_32 = img_encoder_feats[3]
                    feat_camXs_ = feats_8

            else:
                img_encoder_feats = self.encoder(rgb_camXs_)

                feats_4 = img_encoder_feats["feats_4"]
                feats_8 = img_encoder_feats["feats_8"]
                feats_16 = img_encoder_feats["feats_16"]
                feat_camXs_ = img_encoder_feats["output"]

        else:
            feat_camXs_ = self.encoder(rgb_camXs_)["output"]

        if self.use_multi_scale_img_feats:

            if self.rand_flip:
                feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
            feat_camXs = __u(feat_camXs_)  # unpack (B, S, C, Hf, Wf)

        else:
            # "unflip" the image feature maps based on the same random order of the image flipping
            if self.rand_flip:
                feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
            feat_camXs = __u(feat_camXs_)  # (B, S, C, Hf, Wf)

        _, C, Hf, Wf = feat_camXs_.shape  # C=128, Hf=H/8, Wf=W/8

        sy = Hf / float(H)  # sy = 1/8
        sx = Wf / float(W)  # sx = 1/8
        Z_cam, Y_cam, X_cam = self.Z_cam, self.Y_cam, self.X_cam  # 200, 8, 200

        _, _, c_feats, h_feats, w_feats = feat_camXs.shape  # C=128, Hf=H/8, Wf=W/8

        # scale intrinsics
        feat_camXs_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

        if self.xyz_camA is not None:
            # 3d mem in view of reference cam??? (in meters)
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B * S, 1, 1)
        else:
            xyz_camA = None

        # unproject image feats into mem
        cam_feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(feat_camXs_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, self.Z_cam, self.Y_cam, self.X_cam,
            xyz_camA=xyz_camA)

        # unpack features from ([B*S], C, Z, Y, X) -> (B, S, C, Z, Y, X)
        cam_feat_mems = __u(cam_feat_mems_)  # B, S, C, Z, Y, X

        # mask is 1 if abs value is != 0 else zero
        mask_mems = (torch.abs(cam_feat_mems) > 0).float()
        # S = 0 since in 3D feature space we don't need the number of cams dim -> reduce this dim
        cam_feat_mem = utils.basic.reduce_masked_mean(cam_feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        # reshape feats
        cam_feat_bev = cam_feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y_cam, Z_cam, X_cam)

        # first get radar feats from rad encoder
        # bev Enc-Dec
        if self.use_radar:
            assert (rad_occ_mem0 is not None)
            Z_rad, Y_rad, X_rad = self.Z_rad, self.Y_rad, self.X_rad

            # add radar encoding branch
            if self.use_radar_encoder:
                if self.radar_encoder_type == 'voxel_net':
                    rad_bev_ = self.radar_encoder(voxel_features=rad_occ_mem0[0],
                                                  voxel_coords=rad_occ_mem0[1],
                                                  number_of_occupied_voxels=rad_occ_mem0[2],
                                                  dinovoxel=dinovoxel)
                elif self.use_shallow_metadata:
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 4 * Y_rad, Z_rad, X_rad)
                    rad_bev_ = self.radar_encoder(rad_bev_)
                else:
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y_rad, Z_rad, X_rad)
                    rad_bev_ = self.radar_encoder(rad_bev_)
            elif self.use_shallow_metadata and not self.use_radar_encoder:
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 4 * Y_rad, Z_rad, X_rad)  # B,32,200,200
                # for the transformer, we need matching feature dims! --> apply zero-padding here!
                zero_padding = torch.zeros((B, self.latent_dim - (4 * Y_rad), Z_rad, X_rad)).to(device)
                rad_bev_ = torch.cat((rad_bev_, zero_padding), dim=1).to(device)  # C=128

            else:
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y_rad, Z_rad, X_rad)  # C=128

        # Fusion:
        feat_bev_ = torch.cat([cam_feat_bev, rad_bev_], dim=1)  # B,C,200,200

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_bev_[self.bev_flip1_index] = torch.flip(feat_bev_[self.bev_flip1_index], [-1])
            feat_bev_[self.bev_flip2_index] = torch.flip(feat_bev_[self.bev_flip2_index], [-2])

            if rad_occ_mem0 is not None and not (self.radar_encoder_type == "voxel_net"):
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        feat_bev_ = self.bev_compressor(feat_bev_)

        # bev Enc-Dec
        if self.do_feat_enc_dec:
            feat_bev = self.feat_enc_dec(feat_bev_)
        else:
            feat_bev = feat_bev_

        # bev decoder
        seg_e = {}

        if self.train_task == "object":
            out_dict_objects = self.object_decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index)
                                                   if self.rand_flip else None)
            # object estimation data
            obj_seg_e = out_dict_objects['obj_segmentation']
            seg_e = obj_seg_e

        if self.train_task == "map":
            out_dict_map = self.map_decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index)
                                            if self.rand_flip else None)
            # map estimation data
            bev_map_seg_e = out_dict_map['bev_map_segmentation']
            seg_e = bev_map_seg_e

        if self.train_task == "both":
            out_dict_shared = self.shared_decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index)
                                                  if self.rand_flip else None)
            # map estimation data
            bev_map_seg_e = out_dict_shared['bev_map_segmentation']
            obj_seg_e = out_dict_shared['obj_segmentation']
            seg_e = torch.cat([bev_map_seg_e, obj_seg_e], dim=1)  # [b, 8, 200, 200]

        return seg_e
