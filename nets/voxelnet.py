# Concept based on VoxelNet paper: https://arxiv.org/abs/1711.06396
# code based on following unofficial pytorch implementations:
# https://github.com/skyhehe123/VoxelNet-pytorch [1]
# https://github.com/TUMFTM/RadarVoxelFusionNet  [2]
# code modified to fit the desired dimensions use case (radar backbone not a full 3D object detection net
# with respective heads)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from config import config as cfg


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        B, k, t, _ = x.shape
        x = self.linear(x.view(B, k * t, -1))  # -> output: (num_points, num_feature_channels)
        x = F.relu(self.bn(x.permute(0, 2, 1))).permute(0, 2, 1)  # F.relu(self.bn(x.permute(0,2,1)))
        return x.view(B, k, t, -1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(VFE, self).__init__()
        assert out_channels % 2 == 0, 'Number of output channels must be even.'
        self.out_channels = out_channels
        self.half_out_channels = out_channels // 2

        self.fcn = FCN(in_channels, self.half_out_channels)

    def forward(self, x, mask=None):
        """
        Voxel Feature Encoding layer
        :param x: previous layer output
        :param mask: indicating the valid points for further computation (num_voxels, max_points_per_voxel)
        :return: pwcf: 'poit-wise concatenated feature'
        """
        # get shape
        B, num_voxels, max_points_per_voxel, num_features = x.shape  # original implementation only for batch size of 1!

        # point-wise feature
        pwf = self.fcn(x)  # (batch_size x num_voxels x max_points_per_voxel x self.half_out_channels)

        # locally aggregated features, element-wise max-pool
        # (batch_size x num_voxels x 1 x self.half_out_channels)
        laf = torch.max(pwf, dim=2, keepdim=True)[0]

        # Repeat and concatenate with calculated features of each point
        # (batch_size x num_voxels x max_points_per_voxel x self.half_out_channels)
        laf = laf.repeat(1, 1, max_points_per_voxel, 1)

        # point-wise concat feature  -> yields desired output dimension "out_channels" -> concat of 2x out_channels
        # (num_voxels x max_points_per_voxel x self.out_channels)
        pwcf = torch.cat((pwf, laf), dim=-1)

        # apply mask
        # (repeat for number of out_channels per voxel)
        mask = mask.unsqueeze(3).repeat(1, 1, 1, self.out_channels)
        pwcf = pwcf * mask.float()

        return pwcf  # poit-wise concatenated feature


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        # mask of valid points in input
        # (num_voxels, max_points_per_voxel)
        # this masking makes no sense for our application since we only consider valid points as input for voxelnet
        mask = torch.ne(torch.max(x, -1)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # apply mask again to remove invalid features
        mask = mask.unsqueeze(3).repeat(1, 1, 1, x.shape[-1])
        x = x * mask.float()
        # element-wise max pooling
        x = torch.max(x, dim=2)[0]
        return x


# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self, reduced_zx):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        if reduced_zx:
            self.conv3d_3 = Conv3d(64, 64, 3, s=(1, 2, 2), p=(1, 1, 1))
        else:
            self.conv3d_3 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


# OVF - Occupancy Voxel Fuser
class OVF(nn.Module):
    def __init__(self, output_dim):
        super(OVF, self).__init__()
        # create occupancy embedding layer
        self.occ_embedding = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # create voxel feats and occupancy feats fusing layer
        self.occ_voxel_fuser = nn.Sequential(
            nn.Conv2d(2 * output_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, occ, x):
        occ_embed = self.occ_embedding(occ)
        x = torch.cat([x, occ_embed], dim=1)
        x = self.occ_voxel_fuser(x)
        return x


# Region Proposal Network
class RPN(nn.Module):
    def __init__(self, output_dim):
        super(RPN, self).__init__()
        if self.occupancy_radar_rpn:
            # add occupancy layer in RPN
            print("attach occupancy features")
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [Conv2d(256, 256, 3, 1, 1) for _ in
                         range(5)]  # [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))

        # deconv4 for channel reduction and upsampling
        self.deconv_4 = nn.Sequential(nn.ConvTranspose2d(768, output_dim, 2, 2, 0), nn.BatchNorm2d(128))

    def forward(self, x):
        x = self.block_1(x)  # (B,128,200,200) -> (B,128,100,100)
        x_skip_1 = x  # (B,128,100,100)
        x = self.block_2(x)  # (B,128,100,100) -> (B,128,50,50)
        x_skip_2 = x  # (B,128,50,50)
        x = self.block_3(x)  # (B,128,50,50) -> (B,256,25,25)
        x_0 = self.deconv_1(x)  # (B,256,100,100)
        x_1 = self.deconv_2(x_skip_2)  # (B,256,100,100)
        x_2 = self.deconv_3(x_skip_1)  # (B,256,100,100)
        x = torch.cat((x_0, x_1, x_2), 1)  # (B,768,100,100)
        x = self.deconv_4(x)  # (B,128, 200, 200
        x = F.relu(x, inplace=True)
        # adaptation -> add another deconv. layer to get feats down to 128 and shape to 200x200
        return x  # self.score_head(x),self.reg_head(x)


class VoxelNet(nn.Module):

    def __init__(self, use_col=False, reduced_zx=False, output_dim=128, use_radar_occupancy_map=False):
        super(VoxelNet, self).__init__()
        self.use_col = use_col  # use convolutional output layer
        self.reduced_zx = reduced_zx
        self.output_dim = output_dim
        self.svfe = SVFE()
        self.cml = CML(self.reduced_zx)
        self.use_radar_occupancy_map = use_radar_occupancy_map
        if self.output_dim != 128 and not use_col:
            self.fit_adapter = nn.Sequential(
                nn.Conv2d(128, self.output_dim, kernel_size=1, padding=0, stride=1, bias=False),
                nn.InstanceNorm2d(self.output_dim),
                nn.GELU(),
            )
        if use_col:
            self.rpn = RPN(output_dim=self.output_dim)
        if self.use_radar_occupancy_map:
            # add occupancy encoding layers
            self.occ_voxel_fuse = OVF(output_dim=self.output_dim)

    def voxel_indexing(self, sparse_features, coords, number_of_occupied_voxels, dinovoxel=None):
        B, voxels, dim = sparse_features.shape
        dense_feature = Variable(torch.zeros((B, dim, 200, 8, 200), device=sparse_features.device))
        coords_idx = coords.long()
        max_voxels = torch.max(number_of_occupied_voxels).long()

        N = coords_idx.shape[1]
        z = coords_idx[:, :, 0].view(B * N)
        y = coords_idx[:, :, 1].view(B * N)
        x = coords_idx[:, :, 2].view(B * N)
        feat = sparse_features.view(B * N, -1)

        X = 200
        Y = 8
        Z = 200
        D2 = 128

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z
        # dim3, dim2, dim1 = torch.meshgrid(X, X * Y, X * Y * Z)

        base = torch.arange(0, B, dtype=torch.int32, device=sparse_features.device) * dim1
        # base = torch.arange(0, B, dtype=torch.int32, device=sparse_features.device) * dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

        vox_inds = base + z * dim2 + y * dim3 + x

        feat_voxels = torch.zeros((B * Z * Y * X, D2), device=sparse_features.device).float()
        feat_voxels[vox_inds.long()] = feat

        # zero out the singularity
        feat_voxels[base.long()] = 0.0
        feat_voxels = feat_voxels.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)
        # B x C x Z x Y x X

        # new: permute shape such that they fit the 3D Conv layers ZYX -> YZX
        # dense_feature = dense_feature.permute(0, 1, 3, 2, 4)
        occupancy_map = torch.zeros((B * Z * Y * X, D2), device=sparse_features.device).float()
        occupancy_map[vox_inds.long()] = 1.0
        occupancy_map[base.long()] = 0.0
        occupancy_map = occupancy_map.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)
        occupancy_map = torch.sum(occupancy_map, dim=3)  # reduce Y (height)
        occupancy_map = torch.sum(occupancy_map, dim=1).unsqueeze(1)  # reduce D2 to 1

        occupancy_map[occupancy_map > 1] = 1

        if dinovoxel is not None:  # if we get voxelized dino features
            # combines dino and VoxelNet features --> fill empty voxels with dino information
            # the simplest method: replace voxels in dinovoxel where we have encoded radar information
            dinovoxel_buffer = dinovoxel.permute(0, 2, 3, 4, 1).reshape(B * Z * Y * X, D2)
            temp_storage = dinovoxel_buffer[base.long()]
            dinovoxel_buffer[vox_inds.long()] = feat
            dinovoxel_buffer[base.long()] = temp_storage
            dinovoxel = dinovoxel_buffer.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)
            feat_voxels = dinovoxel

        feat_voxels = feat_voxels.permute(0, 1, 3, 2, 4)
        return feat_voxels, occupancy_map

    def forward(self, voxel_features, voxel_coords, number_of_occupied_voxels, dinovoxel=None):

        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs, radar_occupancy_map = self.voxel_indexing(vwfs, voxel_coords, number_of_occupied_voxels,
                                                        dinovoxel)  # efficient representation of features in bev space

        # convolutional middle network
        cml_out = self.cml(vwfs)

        # region proposal network
        # merge the depth and feature dim into one, output probability score map and regression map
        # psm, rm = self.rpn(cml_out.view(config.N, -1, config.H, config.W))  # cfg.N,-1,cfg.H, cfg.W))
        # reshape for rpn input
        cml_out = cml_out.view(cml_out.shape[0], -1, cml_out.shape[-2], cml_out.shape[-1])
        if self.use_col:  # if using rpn layers
            radar_feats = self.rpn(cml_out)  # cfg.N,-1,cfg.H, cfg.W))
        else:
            if self.output_dim != 128 and False:
                cml_out = self.fit_adapter(cml_out)
            cml_out_dims = cml_out.clone().requires_grad_(True)
            radar_feats = cml_out_dims

        if self.use_radar_occupancy_map:
            radar_feats = self.occ_voxel_fuse(occ=radar_occupancy_map, x=radar_feats)

        return radar_feats
