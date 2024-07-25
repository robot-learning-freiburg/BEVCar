# adapeted from: https://github.com/aharley/simple_bev/blob/main/utils/vox.py
import numpy as np
import torch
import torch.nn.functional as F

import utils.geom


class Vox_util(object):
    def __init__(self, Z, Y, X, scene_centroid, bounds, pad=None, assert_cube=False):
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        B, D = list(scene_centroid.shape)
        self.Z, self.Y, self.X = Z, Y, X

        scene_centroid = scene_centroid.detach().cpu().numpy()
        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid

        self.default_vox_size_X = (self.XMAX - self.XMIN) / float(X)
        self.default_vox_size_Y = (self.YMAX - self.YMIN) / float(Y)
        self.default_vox_size_Z = (self.ZMAX - self.ZMIN) / float(Z)

        if pad:
            Z_pad, Y_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)) or (
            not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                      )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert (np.isclose(self.default_vox_size_X, self.default_vox_size_Y))
            assert (np.isclose(self.default_vox_size_X, self.default_vox_size_Z))

    def Ref2Mem(self, xyz, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        # transforms ref coordinates into mem coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert (C == 3)
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        """
        mem_T_ref:
        tensor([[
         [ 2.0000,  0.0000,  0.0000, 99.5000],
         [ 0.0000,  0.8000,  0.0000,  2.7000],
         [ 0.0000,  0.0000,  2.0000, 99.5000],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
        """

        xyz = utils.geom.apply_4x4(mem_T_ref, xyz)
        return xyz

    def Mem2Ref(self, xyz_mem, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube, device=xyz_mem.device)
        xyz_ref = utils.geom.apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_mem_T_ref(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        vox_size_X = (self.XMAX - self.XMIN) / float(X)  # 0.5
        vox_size_Y = (self.YMAX - self.YMIN) / float(Y)  # 1.25
        vox_size_Z = (self.ZMAX - self.ZMIN) / float(Z)  # 0.5

        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                      )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert (np.isclose(vox_size_X, vox_size_Y))
            assert (np.isclose(vox_size_X, vox_size_Z))

        # translation
        # (this makes the left edge of the leftmost voxel correspond to XMIN)
        center_T_ref = utils.geom.eye_4x4(B, device=device)
        center_T_ref[:, 0, 3] = -self.XMIN - vox_size_X / 2.0
        center_T_ref[:, 1, 3] = -self.YMIN - vox_size_Y / 2.0
        center_T_ref[:, 2, 3] = -self.ZMIN - vox_size_Z / 2.0
        """
        center_T_ref:
        tensor([[
         [ 1.0000,  0.0000,  0.0000, 49.7500],
         [ 0.0000,  1.0000,  0.0000,  3.3750],
         [ 0.0000,  0.0000,  1.0000, 49.7500],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
        """

        # scaling
        # (this makes the right edge of the rightmost voxel correspond to XMAX)
        mem_T_center = utils.geom.eye_4x4(B, device=device)
        mem_T_center[:, 0, 0] = 1. / vox_size_X
        mem_T_center[:, 1, 1] = 1. / vox_size_Y
        mem_T_center[:, 2, 2] = 1. / vox_size_Z
        mem_T_ref = utils.basic.matmul2(mem_T_center, center_T_ref)

        """
        mem_T_center:
        tensor([[
         [2.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.8000, 0.0000, 0.0000],
         [0.0000, 0.0000, 2.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0')

        mem_T_ref:
        tensor([[
         [ 2.0000,  0.0000,  0.0000, 99.5000],
         [ 0.0000,  0.8000,  0.0000,  2.7000],
         [ 0.0000,  0.0000,  2.0000, 99.5000],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
        """

        return mem_T_ref

    def get_ref_T_mem(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_inbounds(self, xyz, Z, Y, X, already_mem=False, padding=0.0, assert_cube=False):
        # xyz is B x N x 3
        # padding should be 0 unless you are trying to account for some later cropping
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X, assert_cube=assert_cube)

        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]

        x_valid = ((x - padding) > -0.5).byte() & ((x + padding) < float(X - 0.5)).byte()
        y_valid = ((y - padding) > -0.5).byte() & ((y + padding) < float(Y - 0.5)).byte()
        z_valid = ((z - padding) > -0.5).byte() & ((z + padding) < float(Z - 0.5)).byte()
        nonzero = (~(z == 0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def voxelize_xyz(self, xyz_ref, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert (D == 3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:, 0:1] * 0, Z, Y, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return vox

    def voxelize_xyz_and_feats(self, xyz_ref, feats, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)  # point coords: B, N=num_of_points, D=3
        B2, N2, D2 = list(feats.shape)  # point features: B, N2=num_of_points, D2=16 -> dims of metadata
        assert (D == 3)
        assert (B == B2)
        assert (N == N2)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            # transform from measurement coords to memory coords
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)

            xyz_zero = self.Ref2Mem(xyz_ref[:, 0:1] * 0, Z, Y, X, assert_cube=assert_cube)
        feats = self.get_feat_occupancy(xyz_mem, feats, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return feats

    def voxelize_xyz_and_feats_voxelnet(self, xyz_ref, feats, Z, Y, X, already_mem=False, assert_cube=False,
                                        clean_eps=0, use_radar_occupancy_map=False):
        B, N, D = list(xyz_ref.shape)  # point coords: B, N=num_of_points, D=3
        B2, N2, D2 = list(feats.shape)  # point features: B, N2=num_of_points, D2=16 -> dims of metadata
        assert (D == 3)
        assert (B == B2)
        assert (N == N2)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            # transform from measurement coords to memory coords
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)

            xyz_zero = self.Ref2Mem(xyz_ref[:, 0:1] * 0, Z, Y, X, assert_cube=assert_cube)
        # feats = self.get_feat_occupancy(xyz_mem, feats, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)

        is_voxelnet = True
        if is_voxelnet:
            voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels = \
                self.get_voxelnet_dense_feature_voxels(xyz_mem, feats, Z, Y, X, clean_eps=clean_eps,
                                                       xyz_zero=xyz_zero,
                                                       use_radar_occupancy_map=use_radar_occupancy_map)

            return voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels

        is_pointnet = False
        if is_pointnet:
            radar_point_cloud_feats_filtered = self.get_voxelnet_dense_feature_voxels(
                xyz_mem, feats, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)

            return radar_point_cloud_feats_filtered

    def get_occupancy(self, xyz, Z, Y, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        assert (C == 3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero - xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz)  # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x * mask
        y = y * mask
        z = z * mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X - 1).int()
        y = torch.clamp(y, 0, Y - 1).int()
        z = torch.clamp(z, 0, Z - 1).int()

        x = x.view(B * N)
        y = y.view(B * N)
        z = z.view(B * N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device) * dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B * Z * Y * X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        # zero out the singularity
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Z, Y, X)
        # B x 1 x Z x Y x X
        return voxels

    def get_feat_occupancy(self, xyz, feat, Z, Y, X, clean_eps=0, xyz_zero=None, is_voxelnet=False):
        # xyz is B x N x 3 and in mem coords
        # feat is B x N x D
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        B2, N2, D2 = list(feat.shape)
        assert (C == 3)
        assert (B == B2)
        assert (N == N2)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero  -> of 0.1 ?!
            dist = torch.norm(xyz_zero - xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz)  # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        dummy_coords = torch.stack((x[:, -1], y[:, -1], z[:, -1]), dim=1)
        dummy_coords_reshape = dummy_coords.unsqueeze(dim=1)
        mask = torch.where(xyz == dummy_coords_reshape, 0.0, mask.unsqueeze(-1))
        # mask = torch.where(xyz == dummy_coords, 0.0, mask.unsqueeze(-1))  # throws error due to shape mismatch if B>1
        mask = mask[:, :, 0]

        # (this method seems a bit clumsy)
        x = x * mask  # B, N
        y = y * mask
        z = z * mask
        # add mask as an additional feature
        use_occupancy_feature = True
        if use_occupancy_feature:
            feat = torch.cat((feat, mask.unsqueeze(-1)), dim=2)
        feat = feat * mask.unsqueeze(-1)  # B, N, D=16

        # New store radar points in voxels
        if is_voxelnet:
            """ Work in Progress """
            # ############################# START OF VOXELNET PREPROCESSING ################################################
            # like in VoxelNet  -> define tensor KxTx7  -> BxKxTx19  K: number of maximum occupied voxels here: 3500
            # T=10
            voxel_input_buffer = torch.zeros((B, 3500, 10, 3 + feat.shape[-1]), device=x.device)
            voxel_coord_buffer = torch.zeros((B, 3500, 3), device=x.device)

            # get last element which is the transformed placeholder element:
            dummy_coords = torch.stack((x[:, -1], y[:, -1], z[:, -1]), dim=1)

            # randomize
            perm = torch.randperm(N)
            x = x[:, perm]
            y = y[:, perm]
            z = z[:, perm]
            feat = feat[:, perm]

            radar_point_cloud = torch.zeros_like(xyz)
            radar_point_cloud[:, :, 0] = x
            radar_point_cloud[:, :, 1] = y
            radar_point_cloud[:, :, 2] = z

            voxel_indices = radar_point_cloud.float().long()

            voxel_dict_b = {}
            number_off_occupied_voxels = torch.zeros((B), device=x.device)
            for b in range(B):
                voxel_dict = {}
                voxel_k_dict = {}
                k = 0
                t = 0
                k_coord_buf = 0
                for i, voxel_idx in enumerate(voxel_indices[b, :, :]):
                    voxel_idx = tuple(voxel_idx.tolist())
                    query_point = torch.tensor((x[b, i], y[b, i], z[b, i]), device=x.device).unsqueeze(dim=0)
                    if voxel_idx in voxel_dict and not \
                            all([v_idx == 0 for v_idx in voxel_idx]) and not \
                            torch.eq(query_point, dummy_coords[b]).all().item() and voxel_k_dict[voxel_idx][1] < 9:
                        # add to existing points
                        voxel_dict[voxel_idx].append(i)
                        k, t = voxel_k_dict[voxel_idx]
                        x_t = torch.tensor(x[b, i]).unsqueeze(dim=0)
                        y_t = torch.tensor(y[b, i]).unsqueeze(dim=0)
                        z_t = torch.tensor(z[b, i]).unsqueeze(dim=0)
                        feat_t = torch.tensor(feat[b, i])
                        data = torch.cat([x_t, y_t, z_t, feat_t])
                        voxel_input_buffer[b, k, t + 1, :] = data
                        voxel_k_dict[voxel_idx] = k, t + 1

                        # voxel_input_buffer[b, k, t+1, :] = [x, y, z, feat]
                        # voxel_k_dict[voxel_idx] = k, t+1
                    elif not all([v_idx == 0 for v_idx in voxel_idx]) and not \
                            torch.eq(query_point, dummy_coords[b]).all().item():
                        # add new key
                        voxel_dict[voxel_idx] = [i]

                        voxel_coord_buffer[b, k_coord_buf, :] = torch.tensor(voxel_idx)
                        x_t = torch.tensor(x[b, i]).unsqueeze(dim=0)
                        y_t = torch.tensor(y[b, i]).unsqueeze(dim=0)
                        z_t = torch.tensor(z[b, i]).unsqueeze(dim=0)
                        feat_t = torch.tensor(feat[b, i])
                        data = torch.cat([x_t, y_t, z_t, feat_t])
                        voxel_input_buffer[b, k_coord_buf, t, :] = data

                        # voxel_coord_buffer[b, k_coord_buf, :] = voxel_idx
                        # voxel_input_buffer[b, k_coord_buf, t, :] = [x, y, z, feat]

                        voxel_k_dict[voxel_idx] = [k_coord_buf, t]
                        k_coord_buf += 1
                        # t += 1

                # voxel_dict_b.append(voxel_dict)
                return voxel_input_buffer, voxel_coord_buffer

            # ############################# END OF VOXELNET PREPROCESSING ##################################################

        # old code
        x = torch.round(x)  # rounding to the next integer -> thats where voxelnet should grab the un-rounded points
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X - 1).int()
        y = torch.clamp(y, 0, Y - 1).int()
        z = torch.clamp(z, 0, Z - 1).int()

        # permute point orders
        perm = torch.randperm(N)
        x = x[:, perm]
        y = y[:, perm]
        z = z[:, perm]
        feat = feat[:, perm]

        x = x.view(B * N)
        y = y.view(B * N)
        z = z.view(B * N)
        feat = feat.view(B * N, -1)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device) * dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

        vox_inds = base + z * dim2 + y * dim3 + x
        # feat_voxels = torch.zeros((B*Z*Y*X, D2), device=xyz.device).float()
        # +1 since we added the occupancy feature layer
        feat_voxels = torch.zeros((B * Z * Y * X, D2 + 1), device=xyz.device).float()
        feat_voxels[vox_inds.long()] = feat
        # zero out the singularity
        feat_voxels[base.long()] = 0.0
        # feat_voxels = feat_voxels.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)
        # +1 since we added the occupancy feature layer
        feat_voxels = feat_voxels.reshape(B, Z, Y, X, D2 + 1).permute(0, 4, 1, 2, 3)
        # B x C x Z x Y x X
        return feat_voxels

    def get_voxelnet_dense_feature_voxels(self, xyz, feat, Z, Y, X, clean_eps=0, xyz_zero=None,
                                          use_radar_occupancy_map=False):
        # xyz is B x N x 3 and in mem coords
        # feat is B x N x D
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        B2, N2, D2 = list(feat.shape)
        assert (C == 3)
        assert (B == B2)
        assert (N == N2)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero  -> of 0.1 ?!  -> changed to 0.3
            dist = torch.norm(xyz_zero - xyz, dim=2)
            mask[dist < 0.1] = 0  # default: 0.1

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz)  # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # remove dummy coords from data for pointnet
        dummy_coords = torch.stack((x[:, -1], y[:, -1], z[:, -1]), dim=1)
        dummy_coords_reshape = dummy_coords.unsqueeze(dim=1)
        mask = torch.where(xyz == dummy_coords_reshape, 0.0, mask.unsqueeze(-1))
        mask = mask[:, :, 0]
        xyz_zero_prefiltered = torch.where(xyz == dummy_coords, 0.0, xyz)
        xyz_zero_prefiltered_np = xyz_zero_prefiltered.detach().cpu().numpy()
        x, y, z = xyz_zero_prefiltered[:, :, 0], xyz_zero_prefiltered[:, :, 1], xyz_zero_prefiltered[:, :, 2]

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x * mask  # B, N
        y = y * mask
        z = z * mask

        # add mask as an additional feature
        feat = torch.cat((feat, mask.unsqueeze(-1)), dim=2)
        feat = feat * mask.unsqueeze(-1)  # B, N, D=16

        # for pointnet
        radar_point_cloud_filtered = torch.zeros_like(xyz)
        radar_point_cloud_filtered[:, :, 0] = x
        radar_point_cloud_filtered[:, :, 1] = y
        radar_point_cloud_filtered[:, :, 2] = z

        # ZYX fromat  -> actually correct
        # radar_point_cloud_filtered[:, :, 0] = z
        # radar_point_cloud_filtered[:, :, 1] = y
        # radar_point_cloud_filtered[:, :, 2] = x

        radar_point_cloud_feats_filtered = torch.cat([radar_point_cloud_filtered, feat], dim=2)
        radar_point_cloud_feats_filtered_np = radar_point_cloud_feats_filtered.detach().cpu().numpy()

        is_pointnet = False
        if is_pointnet:
            return radar_point_cloud_feats_filtered
        # New store radar points in voxels

        # ############################# START OF VOXELNET PREPROCESSING ################################################
        # like in VoxelNet  -> define tensor KxTx7  -> BxKxTx19  K: number of maximum occupied voxels here: 3500
        # T=10
        voxel_input_buffer = torch.zeros((B, 3500, 10, 3 + feat.shape[-1]), device=x.device)
        voxel_coord_buffer = torch.zeros((B, 3500, 3), device=x.device)

        # get last element which is the transformed placeholder element:
        dummy_coords = torch.stack((x[:, -1], y[:, -1], z[:, -1]), dim=1)
        # randomize
        perm = torch.randperm(N)
        x = x[:, perm]
        y = y[:, perm]
        z = z[:, perm]
        feat = feat[:, perm]

        radar_point_cloud = torch.zeros_like(xyz)
        # radar_point_cloud[:, :, 0] = x
        # radar_point_cloud[:, :, 1] = y
        # radar_point_cloud[:, :, 2] = z
        # in ZYX Format
        radar_point_cloud[:, :, 0] = z
        radar_point_cloud[:, :, 1] = y
        radar_point_cloud[:, :, 2] = x

        voxel_indices = torch.round(radar_point_cloud).int()
        # clamp values so that they do not exceed the valid boundaries
        voxel_indices[:, :, 0] = torch.clamp(voxel_indices[:, :, 0], 0, Z - 1).int()
        voxel_indices[:, :, 1] = torch.clamp(voxel_indices[:, :, 1], 0, Y - 1).int()
        voxel_indices[:, :, 2] = torch.clamp(voxel_indices[:, :, 2], 0, X - 1).int()

        voxel_dict_b = {}

        number_off_occupied_voxels = torch.zeros(B, device=x.device)

        for b in range(B):
            voxel_dict = {}
            voxel_k_dict = {}
            k = 0
            t = 0
            k_coord_buf = 0
            for i, voxel_idx in enumerate(voxel_indices[b, :, :]):
                voxel_idx = tuple(voxel_idx.tolist())
                query_point = torch.tensor((x[b, i], y[b, i], z[b, i]), device=x.device).unsqueeze(dim=0)
                if voxel_idx in voxel_dict and not \
                        all([v_idx == 0 for v_idx in voxel_idx]) and not \
                        torch.eq(query_point, dummy_coords[b]).all().item() and voxel_k_dict[voxel_idx][1] < 9:
                    # add to existing points
                    voxel_dict[voxel_idx].append(i)
                    k, t = voxel_k_dict[voxel_idx]

                    x_t = x[b, i].clone().detach().unsqueeze(dim=0)
                    y_t = y[b, i].clone().detach().unsqueeze(dim=0)
                    z_t = z[b, i].clone().detach().unsqueeze(dim=0)
                    feat_t = feat[b, i].clone().detach()
                    # Z,Y,X order
                    data = torch.cat([z_t, y_t, x_t, feat_t])
                    voxel_input_buffer[b, k, t + 1, :] = data
                    voxel_k_dict[voxel_idx] = k, t + 1

                    # debug:
                    # voxel_input_buffer_np = voxel_input_buffer.detach().cpu().numpy()

                elif not all([v_idx == 0 for v_idx in voxel_idx]) and not \
                        torch.eq(query_point, dummy_coords[b]).all().item():
                    # add new key
                    voxel_dict[voxel_idx] = [i]
                    voxel_coord_buffer[b, k_coord_buf, :] = torch.tensor(voxel_idx)

                    # fix: "it is recommended to use sourceTensor.clone().detach()"
                    x_t = x[b, i].clone().detach().unsqueeze(dim=0)
                    y_t = y[b, i].clone().detach().unsqueeze(dim=0)
                    z_t = z[b, i].clone().detach().unsqueeze(dim=0)
                    feat_t = feat[b, i].clone().detach()

                    # Z,Y,X order
                    data = torch.cat([z_t, y_t, x_t, feat_t])
                    voxel_input_buffer[b, k_coord_buf, 0, :] = data
                    voxel_k_dict[voxel_idx] = [k_coord_buf, 0]
                    k_coord_buf += 1

                    # debug:
                    # voxel_input_buffer_np = voxel_input_buffer.detach().cpu().numpy()
                    # voxel_coord_buffer_np = voxel_coord_buffer.detach().cpu().numpy()

            number_off_occupied_voxels[b] = k_coord_buf

        # radar_occupancy_map = mask
        # (B, 3500, 10, 19) , (B,3500, 3)
        return voxel_input_buffer, voxel_coord_buffer, number_off_occupied_voxels  # , radar_occupancy_map

    def unproject_image_to_mem(self, rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X, assert_cube=False, xyz_camA=None):
        # rgb_camB is B x C x H x W
        # pixB_T_camA is B x 4 x 4

        # rgb lives in B pixel coords
        # we want everything in A memory coords

        # this puts each C-dim pixel in the rgb_camB
        # along a ray in the voxelgrid
        B, C, H, W = list(rgb_camB.shape)

        if xyz_camA is None:
            xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)
            xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = utils.geom.apply_4x4(camB_T_camA, xyz_camA)
        z = xyz_camB[:, :, 2]

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:, :, 2], 2)
        EPS = 1e-6
        # z = xyz_pixB[:,:,2]
        xy_pixB = xyz_pixB[:, :, :2] / torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:, :, 0], xy_pixB[:, :, 1]
        # these are B x N

        x_valid = (x > -0.5).bool() & (x < float(W - 0.5)).bool()
        y_valid = (y > -0.5).bool() & (y < float(H - 0.5)).bool()
        z_valid = (z > 0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

        if (0):
            # handwritten version
            values = torch.zeros([B, C, Z * Y * X], dtype=torch.float32)
            for b in list(range(B)):
                values[b] = utils.samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
        else:
            # native pytorch version
            y_pixB, x_pixB = utils.basic.normalize_grid2d(y, x, H, W)
            # since we want a 3d output, we need 5d tensors
            z_pixB = torch.zeros_like(x)
            xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
            rgb_camB = rgb_camB.unsqueeze(2)
            xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
            values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False)  # default mode: 'bilinear'

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values

    def warp_tiled_to_mem(self, rgb_tileB, pixB_T_camA, camB_T_camA, Z, Y, X, DMIN, DMAX, assert_cube=False):
        # rgb_tileB is B,C,D,H,W
        # pixB_T_camA is B,4,4
        # camB_T_camA is B,4,4

        # rgb_tileB lives in B pixel coords but it has been tiled across the Z dimension
        # we want everything in A memory coords

        # this resamples the so that each C-dim pixel in rgb_tilB
        # is put into its correct place in the voxelgrid
        # (using the pinhole camera model)

        B, C, D, H, W = list(rgb_tileB.shape)

        xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)

        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = utils.geom.apply_4x4(camB_T_camA, xyz_camA)
        z_camB = xyz_camB[:, :, 2]

        # rgb_tileB has depth=DMIN in tile 0, and depth=DMAX in tile D-1
        z_tileB = (D - 1.0) * (z_camB - float(DMIN)) / float(DMAX - DMIN)

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:, :, 2], 2)
        EPS = 1e-6
        # z = xyz_pixB[:,:,2]
        xy_pixB = xyz_pixB[:, :, :2] / torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:, :, 0], xy_pixB[:, :, 1]
        # these are B x N

        x_valid = (x > -0.5).bool() & (x < float(W - 0.5)).bool()
        y_valid = (y > -0.5).bool() & (y < float(H - 0.5)).bool()
        z_valid = (z_camB > 0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

        z_tileB, y_pixB, x_pixB = utils.basic.normalize_grid3d(z_tileB, y, x, D, H, W)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_tileB], axis=2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_tileB, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values

    def apply_mem_T_ref_to_lrtlist(self, lrtlist_cam, Z, Y, X, assert_cube=False):
        # lrtlist is B x N x 19, in cam coordinates
        # transforms them into mem coordinates, including a scale change for the lengths
        B, N, C = list(lrtlist_cam.shape)
        assert (C == 19)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=lrtlist_cam.device)

    def xyz2circles(self, xyz, radius, Z, Y, X, soft=True, already_mem=True, also_offset=False, grid=None):
        # xyz is B x N x 3
        # radius is B x N or broadcastably so
        # output is B x N x Z x Y x X
        B, N, D = list(xyz.shape)
        assert (D == 3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)

        if grid is None:
            grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X, stack=False, norm=False, device=xyz.device)
            # note the default stack is on -1
            grid = torch.stack([grid_x, grid_y, grid_z], dim=1)
            # this is B x 3 x Z x Y x X

        xyz = xyz.reshape(B, N, 3, 1, 1, 1)
        grid = grid.reshape(B, 1, 3, Z, Y, X)
        # this is B x N x Z x Y x X

        # round the xyzs, so that at least one value matches the grid perfectly,
        # and we get a value of 1 there (since exp(0)==1)
        xyz = xyz.round()

        if torch.is_tensor(radius):
            radius = radius.clamp(min=0.01)

        if soft:
            off = grid - xyz  # B,N,3,Z,Y,X
            # interpret radius as sigma
            dist_grid = torch.sum(off ** 2, dim=2, keepdim=False)
            # this is B x N x Z x Y x X
            if torch.is_tensor(radius):
                radius = radius.reshape(B, N, 1, 1, 1)
            mask = torch.exp(-dist_grid / (2 * radius * radius))
            # zero out near zero
            mask[mask < 0.001] = 0.0
            # h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            # h[h < np.finfo(h.dtype).eps * h.max()] = 0
            # return h
            if also_offset:
                return mask, off
            else:
                return mask
        else:
            assert (False)  # something is wrong with this. come back later to debug

            dist_grid = torch.norm(grid - xyz, dim=2, keepdim=False)
            # this is 0 at/near the xyz, and increases by 1 for each voxel away

            radius = radius.reshape(B, N, 1, 1, 1)

            within_radius_mask = (dist_grid < radius).float()
            within_radius_mask = torch.sum(within_radius_mask, dim=1, keepdim=True).clamp(0, 1)
            return within_radius_mask

    def xyz2circles_bev(self, xyz, radius, Z, Y, X, already_mem=True, also_offset=False):
        # xyz is B x N x 3
        # radius is B x N or broadcastably so
        # output is B x N x Z x Y x X
        B, N, D = list(xyz.shape)
        assert (D == 3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)

        xz = torch.stack([xyz[:, :, 0], xyz[:, :, 2]], dim=2)

        grid_z, grid_x = utils.basic.meshgrid2d(B, Z, X, stack=False, norm=False, device=xyz.device)
        # note the default stack is on -1
        grid = torch.stack([grid_x, grid_z], dim=1)
        # this is B x 2 x Z x X

        xz = xz.reshape(B, N, 2, 1, 1)
        grid = grid.reshape(B, 1, 2, Z, X)
        # these are ready to broadcast to B x N x Z x X

        # round the points, so that at least one value matches the grid perfectly,
        # and we get a value of 1 there (since exp(0)==1)
        xz = xz.round()

        if torch.is_tensor(radius):
            radius = radius.clamp(min=0.01)

        off = grid - xz  # B,N,2,Z,X
        # interpret radius as sigma
        dist_grid = torch.sum(off ** 2, dim=2, keepdim=False)
        # this is B x N x Z x X
        if torch.is_tensor(radius):
            radius = radius.reshape(B, N, 1, 1, 1)
        mask = torch.exp(-dist_grid / (2 * radius * radius))
        # zero out near zero
        mask[mask < 0.001] = 0.0

        # add a Y dim
        mask = mask.unsqueeze(-2)
        off = off.unsqueeze(-2)
        # # B,N,2,Z,1,X

        if also_offset:
            return mask, off
        else:
            return mask
