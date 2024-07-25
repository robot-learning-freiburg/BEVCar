import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import yaml
from shapely.errors import ShapelyDeprecationWarning
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import nuscenes_data
import saverloader
import utils.basic
import utils.geom
import utils.improc
import utils.misc
import utils.vox
import wandb
from nets.segnet_simple_bev_with_map import SegnetWithMap
from nets.segnet_simple_lift_fuse_ablation_new_decoders import (
    SegnetSimpleLiftFuse,
)
from nets.segnet_transformer_lift_fuse_new_decoders import (
    SegnetTransformerLiftFuse,
)

# Suppress deprecation warnings from shapely regarding the nuscenes map api
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning, module="nuscenes.map_expansion.map_api")

torch.multiprocessing.set_sharing_strategy('file_system')

# set seed in the beginning
random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

# the scene centroid is defined wrt a reference camera,
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0  # down 1 meter
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200


def requires_grad(parameters: iter, flag: bool = True) -> None:
    """
    Sets the `requires_grad` attribute of the given parameters.
    Args:
        parameters (iterable): An iterable of parameter tensors whose `requires_grad` attribute will be set.
        flag (bool, optional): If True, sets `requires_grad` to True. If False, sets it to False.
            Default is True.

    Returns:
        None
    """
    for p in parameters:
        p.requires_grad = flag


def fetch_optimizer(lr: float, wdecay: float, epsilon: float, num_steps: int, params: iter) \
        -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Fetches an AdamW optimizer and a OneCycleLR scheduler.
    Args:
        lr (float): Learning rate for the optimizer.
        wdecay (float): Weight decay (L2 penalty) for the optimizer.
        epsilon (float): Term added to the denominator to improve numerical stability in the optimizer.
        num_steps (int): Number of steps for the learning rate scheduler.
        params (iter): Iterable of parameters to optimize or dictionaries defining parameter groups.

    Returns:
        tuple: A tuple containing the optimizer and the learning rate scheduler.
            - optimizer (torch.optim.AdamW): The AdamW optimizer.
            - scheduler (torch.optim.lr_scheduler.OneCycleLR): The OneCycleLR learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps + 100, pct_start=0.05,
                                                    cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class SimpleLoss(torch.nn.Module):
    """
    SimpleLoss module that computes the binary cross-entropy loss.

    Args:
        pos_weight (float): Positive class weight for the binary cross-entropy loss.

    Methods:
        forward(ypred: torch.Tensor, ytgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            Forward pass that computes the binary cross-entropy loss.
    """

    def __init__(self, pos_weight: float):
        """Initializes the SimpleLoss module with the specified positive class weight."""
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred: torch.Tensor, ytgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the binary cross-entropy loss.

        Args:
            ypred (torch.Tensor): Predicted logits.
            ytgt (torch.Tensor): Target tensor.
            valid (torch.Tensor): Mask indicating valid elements.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss


class SigmoidFocalLoss(torch.nn.Module):
    """
    Computes the sigmoid of the model output to get values between 0 and 1, then applies the Focal Loss.
    """

    def __init__(self, alpha: float = -1.0, gamma: int = 2, reduction: str = "mean"):
        """
        Args:
            alpha (float, optional): Balances the importance of positive/negative examples. Default is -1.0.
            gamma (int, optional): If >= 0, reduces the loss contribution from easy examples
                and extends the range in which an example receives low loss. Default is 2.
            reduction (str, optional): Specifies the reduction to apply to the output. Options are 'mean', 'sum',
                and 'sum_of_class_means'. Default is 'mean'.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, map_seg_e: torch.Tensor, map_seg_gt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the sigmoid focal loss.

        Args:
            map_seg_e (torch.Tensor): Predicted logits.
            map_seg_gt (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        # get predictions between 0 and 1
        p = torch.sigmoid(map_seg_e)
        # BCE with logits
        ce_loss = F.binary_cross_entropy_with_logits(input=map_seg_e, target=map_seg_gt, reduction="none")
        p_t = p * map_seg_gt + (1 - p) * (1 - map_seg_gt)
        f_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * map_seg_gt + (1 - self.alpha) * (1 - map_seg_gt)
            f_loss = alpha_t * f_loss
        else:
            f_loss = f_loss

        if self.reduction == "mean":  # get mean over all classes
            f_loss = f_loss.mean()
        elif self.reduction == "sum":
            f_loss = f_loss.sum()
        elif self.reduction == "sum_of_class_means":
            # mean over B and bev grid -> then sum avg class error
            f_loss = f_loss.mean(dim=[0, 2, 3]).sum()
        return f_loss


def grad_acc_metrics(metrics_single_pass: dict, metrics_mean_grad_acc: dict, internal_step: int, grad_acc: int) \
        -> dict:
    """
    Accumulates metrics over gradient accumulation steps and computes mean values.
    Args:
        metrics_single_pass (dict): Dictionary containing metrics for a single pass.
        metrics_mean_grad_acc (dict): Dictionary containing accumulated metrics over gradient accumulation steps.
        internal_step (int): Current internal step within the gradient accumulation process.
        grad_acc (int): Number of gradient accumulation steps.

    Returns:
        dict: Dictionary containing mean values of accumulated metrics.
    """
    # Idea: loop over all keys -> if value is None -> do nothing; if value is not None -> accumulate
    for key in metrics_single_pass.keys():
        if metrics_single_pass[key] is not None and key != 'map_seg_thresholds':
            metrics_mean_grad_acc[key] += metrics_single_pass[key]
        else:
            metrics_mean_grad_acc[key] = metrics_single_pass[key]
    # Calculate mean values for losses, but accumulate intersections and unions, no early mean computation
    if internal_step == grad_acc - 1:
        for key in metrics_mean_grad_acc.keys():
            if metrics_mean_grad_acc[key] is not None:  # Exclude mean over intersections/unions
                if key not in ['obj_intersections', 'obj_unions', 'map_masks_intersections', 'map_masks_unions',
                               'map_masks_multi_ious_intersections', 'map_masks_multi_ious_unions',
                               'map_seg_thresholds']:
                    metrics_mean_grad_acc[key] = metrics_mean_grad_acc[key] / grad_acc  # Calculate mean
            else:
                metrics_mean_grad_acc[key] = None
    return metrics_mean_grad_acc


def collect_metrics_for_wandb(total_loss: torch.Tensor, metrics: dict, mode: str, pool_dict: dict,
                              pool_name: str, commit: bool = False, train_task: str = 'both') -> None:
    """
    Function for handling most of the metric and other data collection for wandb.

    Args:
        total_loss (torch.Tensor): Combined loss directly received as model output.
        metrics (dict): Dictionary holding all other metrics.
        mode (str): Specifies whether the collected data belongs to:
            'train' for normal training run,
            'train_ddp' for training in a DDP setup.
        pool_dict (dict): Dictionary holding all pooling objects.
        pool_name (str): Name of the pooling dict.
        commit (bool, optional): If True, data will be sent to wandb immediately. Default is False.
        train_task (str, optional): Specifies the logging task with respect to the respective tasks:
            'both', 'object', or 'map'. Default is 'both'.

    Returns:
        None

    """

    map_labels = ['drivable', 'carpark', 'ped_cross', 'walkway', 'stop_line', 'road_divider',
                  'lane_divider']
    iou_thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    ddp_train_metrics = {}
    ddp_train_metrics_object = {}
    ddp_train_metrics_map = {}

    if mode == 'train_ddp':
        # total loss
        pool_dict['loss_pool_' + pool_name].update([total_loss.item()])
        ddp_train_metrics['pooled/total_loss'] = pool_dict['loss_pool_' + pool_name].mean()
        ddp_train_metrics['stats/total_loss'] = total_loss
        # log loss metrics
        wandb.log({'DDP_train_metrics': ddp_train_metrics}, commit=commit)

        if train_task == 'both' or train_task == 'object':
            # object seg iou
            pool_dict['obj_iou_pool_' + pool_name].update([metrics['obj_iou'].item()])
            # object seg errors
            pool_dict['ce_pool_' + pool_name].update([metrics['ce_loss'].item()])
            pool_dict['ce_weight_pool_' + pool_name].update([metrics['ce_weight']])

            ddp_train_metrics_object.update({
                # object seg iou
                'pooled/obj_iou': pool_dict['obj_iou_pool_' + pool_name].mean(),
                'stats/obj_iou': metrics['obj_iou'],
                # object seg errors
                'pooled/ce_loss': pool_dict['ce_pool_' + pool_name].mean(),
                'stats/ce_loss': metrics['ce_loss'],
                # weights
                'pooled/ce_weight': pool_dict['ce_weight_pool_' + pool_name].mean(),
                'stats/ce_weight': metrics['ce_weight'],
            })
            # log object metrics
            wandb.log({'DDP_train_metrics_object': ddp_train_metrics_object}, commit=commit)

        if train_task == 'both' or train_task == 'map':
            # update map pool_dict
            for label in map_labels:
                pool_key = f'{label}_iou_pool_{pool_name}'
                metric_key = f'{label}_iou'
                pool_dict[pool_key].update([metrics[metric_key]])

            # mean iou over map masks
            pool_dict['masks_mean_iou_pool_' + pool_name].update([metrics['masks_mean_iou']])
            # map seg error
            pool_dict['fc_pool_' + pool_name].update([metrics['focal_loss_map'].item()])
            pool_dict['fc_map_weight_pool_' + pool_name].update([metrics['fc_map_weight']])

            # Loop over all map classes
            for label in map_labels:
                # handle pooled IOUs
                key_pooled_iou = f'pooled/{label}_iou'
                value_pooled_iou = pool_dict[f'{label}_iou_pool_' + pool_name].mean()
                ddp_train_metrics_map[key_pooled_iou] = value_pooled_iou

                # handle stats IOUs
                key_stats_iou = f'stats/{label}_iou'
                value_stats_iou = metrics[f'{label}_iou']
                ddp_train_metrics_map[key_stats_iou] = value_stats_iou

                # collect threshold specific ious
                for threshold in iou_thresholds:
                    key_label_thresh = f'stats/{label}_ious/{threshold}'
                    value_label_thresh = metrics[f'{label}_ious'][iou_thresholds.index(threshold)]
                    ddp_train_metrics_map[key_label_thresh] = value_label_thresh

                # best mean ious per map class
                key_best_ious = f'stats/best_masks_ious/{label}_iou'
                value_best_ious = metrics[f'best_{label}_iou']
                ddp_train_metrics_map[key_best_ious] = value_best_ious

            # handle the best thresholds per class
            for idx, label in enumerate(map_labels):
                key = f'stats/best_thresholds_per_class/{label}_iou'
                value = metrics['best_thresholds'][idx]
                ddp_train_metrics_map[key] = value

            ddp_train_metrics_map.update({
                'pooled/masks_mean_iou': pool_dict['masks_mean_iou_pool_' + pool_name].mean(),
                'stats/masks_mean_iou': metrics['masks_mean_iou'],
                'stats/best_masks_miou': metrics['best_map_mean_iou'].item(),
                'pooled/fc_loss_map': pool_dict['fc_pool_' + pool_name].mean(),
                'stats/fc_loss_map': metrics['focal_loss_map'],
                'pooled/fc_map_weight': pool_dict['fc_map_weight_pool_' + pool_name].mean(),
                'stats/fc_map_weight': metrics['fc_map_weight']
            })
            # log map metrics
            wandb.log({'DDP_train_metrics_map': ddp_train_metrics_map}, commit=commit)


def create_train_pool_dict(name: str, n_pool: int) -> tuple[dict, str]:
    """
    Creates a dictionary of training pools for tracking various metrics during training.

    Args:
        name (str): Name suffix for the pool dictionary keys.
        n_pool (int): Number of values included for the moving average.

    Returns:
        tuple[dict, str]: A tuple containing the dictionary of training pools and the name suffix.
            The dictionary includes pools for:
            - Total loss
            - Time
            - Object segmentation IoU
            - Map masks IoU for various classes (drivable area, carpark, pedestrian crossing, etc.)
            - Mean IoU for map masks
            - Specific losses for object and map segmentation
    """

    train_pool_dict = {
        # total loss
        'loss_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # time
        'time_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # object segmentation IoU
        'obj_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),

        # map masks
        'drivable_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'carpark_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'ped_cross_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'walkway_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'stop_line_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'road_divider_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'lane_divider_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # mean map maks iou
        'masks_mean_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),

        # specific losses
        # object seg
        'ce_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'ce_weight_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # map seg
        'fc_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'fc_map_weight_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
    }
    return train_pool_dict, name


def reduce_loss_metrics(total_loss: torch.Tensor, metrics: dict, train_task: str = 'both') -> tuple[torch.Tensor, dict]:
    """
    combines the loss and the metrics across all batches and GPUs to get the mean performance across the whole dataset
    *sdt : single device tensor
    *adt : all devices' tensor

    Args:
        total_loss (torch.Tensor): total loss returned by the device specific model
        metrics (dict): metrics returned by the device specific model
        train_task (str): 'both', 'object' or 'map' -> enables control on the logging w.r.t. the respective tasks

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing the reduced loss across all batches and GPUs as well as
        the combined metrics
    """

    metrics_reduced = {}

    # get total loss
    total_loss_sdt = torch.tensor([total_loss.item()]).cuda()
    dist.all_reduce(total_loss_sdt, op=dist.ReduceOp.SUM)
    total_loss_adt = total_loss_sdt / dist.get_world_size()
    # total reduced loss
    t_loss_reduced = total_loss_adt

    if train_task == 'both' or train_task == 'map':
        map_seg_thresholds = metrics['map_seg_thresholds']
        metrics_reduced['fc_map_weight'] = metrics['fc_map_weight']

        focal_loss_map_sdt = metrics['focal_loss_map'].clone().detach().cuda()
        dist.all_reduce(focal_loss_map_sdt, op=dist.ReduceOp.SUM)
        focal_loss_map_adt = focal_loss_map_sdt / dist.get_world_size()
        metrics_reduced['focal_loss_map'] = focal_loss_map_adt
        # 7
        map_masks_intersections_sdt = metrics['map_masks_intersections'].clone().detach().cuda()
        map_masks_unions_sdt = metrics['map_masks_unions'].clone().detach().cuda()
        single_thresh_map_metrics_std = torch.stack((map_masks_intersections_sdt, map_masks_unions_sdt))
        dist.all_reduce(single_thresh_map_metrics_std, op=dist.ReduceOp.SUM)
        map_masks_iou_adt = single_thresh_map_metrics_std[-2] / (single_thresh_map_metrics_std[-1] + 1e-4)  # 7
        map_mean_iou_adt = map_masks_iou_adt.sum(dim=0) / torch.count_nonzero(map_masks_iou_adt, dim=0)

        metrics_reduced['drivable_iou'] = map_masks_iou_adt[0].item()
        metrics_reduced['carpark_iou'] = map_masks_iou_adt[1].item()
        metrics_reduced['ped_cross_iou'] = map_masks_iou_adt[2].item()
        metrics_reduced['walkway_iou'] = map_masks_iou_adt[3].item()
        metrics_reduced['stop_line_iou'] = map_masks_iou_adt[4].item()
        metrics_reduced['road_divider_iou'] = map_masks_iou_adt[5].item()
        metrics_reduced['lane_divider_iou'] = map_masks_iou_adt[6].item()

        metrics_reduced['masks_mean_iou'] = map_mean_iou_adt.item()

        # 7,12
        map_masks_multi_ious_intersections_sdt = metrics['map_masks_multi_ious_intersections'].clone().detach().cuda()
        map_masks_multi_ious_unions_sdt = metrics['map_masks_multi_ious_unions'].clone().detach().cuda()
        multi_thresh_map_metrics_std = torch.stack((map_masks_multi_ious_intersections_sdt,
                                                    map_masks_multi_ious_unions_sdt))

        dist.all_reduce(multi_thresh_map_metrics_std, op=dist.ReduceOp.SUM)
        map_masks_multi_iou_adt = multi_thresh_map_metrics_std[-2] / (multi_thresh_map_metrics_std[-1] + 1e-4)  # 7,12
        best_map_ious, best_threshold_index = torch.max(map_masks_multi_iou_adt, dim=1)
        best_map_mean_iou = best_map_ious.sum(dim=0) / torch.count_nonzero(best_map_ious, dim=0)
        best_thresholds = map_seg_thresholds[best_threshold_index]

        metrics_reduced['drivable_ious'] = map_masks_multi_iou_adt[0]  # (1,12) tensor for all threshs
        metrics_reduced['carpark_ious'] = map_masks_multi_iou_adt[1]
        metrics_reduced['ped_cross_ious'] = map_masks_multi_iou_adt[2]
        metrics_reduced['walkway_ious'] = map_masks_multi_iou_adt[3]
        metrics_reduced['stop_line_ious'] = map_masks_multi_iou_adt[4]
        metrics_reduced['road_divider_ious'] = map_masks_multi_iou_adt[5]
        metrics_reduced['lane_divider_ious'] = map_masks_multi_iou_adt[6]

        metrics_reduced['best_drivable_iou'] = best_map_ious[0]
        metrics_reduced['best_carpark_iou'] = best_map_ious[1]
        metrics_reduced['best_ped_cross_iou'] = best_map_ious[2]
        metrics_reduced['best_walkway_iou'] = best_map_ious[3]
        metrics_reduced['best_stop_line_iou'] = best_map_ious[4]
        metrics_reduced['best_road_divider_iou'] = best_map_ious[5]
        metrics_reduced['best_lane_divider_iou'] = best_map_ious[6]

        metrics_reduced['best_map_mean_iou'] = best_map_mean_iou
        metrics_reduced['best_thresholds'] = best_thresholds

    if train_task == 'both' or train_task == 'object':
        metrics_reduced['ce_weight'] = metrics['ce_weight']

        ce_loss_sdt = metrics['ce_loss'].clone().detach().cuda()
        obj_intersections_sdt = metrics['obj_intersections'].clone().detach().cuda()
        obj_unions_sdt = metrics['obj_unions'].clone().detach().cuda()
        single_tensor_obj_metrics_std = torch.stack((ce_loss_sdt, obj_intersections_sdt, obj_unions_sdt))
        # communicate metrics across GPUs
        dist.all_reduce(single_tensor_obj_metrics_std, op=dist.ReduceOp.SUM)
        # Obj. IoU calc
        obj_iou_adt = single_tensor_obj_metrics_std[-2] / (single_tensor_obj_metrics_std[-1] + 1e-4)
        ce_loss_adt = single_tensor_obj_metrics_std[0] / dist.get_world_size()
        metrics_reduced['obj_iou'] = obj_iou_adt
        # single losses
        metrics_reduced['ce_loss'] = ce_loss_adt
    return t_loss_reduced, metrics_reduced


def run_model(model, loss_fn, map_seg_loss_fn, d, device, sw=None, use_radar_encoder=None,
              radar_encoder_type=None, train_task='both', is_master=False, use_shallow_metadata=True,
              use_obj_layer_only_on_map=False):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    voxel_input_feature_buffer = None
    voxel_coordinate_buffer = None
    number_of_occupied_voxels = None
    in_occ_mem0 = None

    if radar_encoder_type == "voxel_net":
        # voxelnet
        imgs, rots, trans, intrins, seg_bev_g, valid_bev_g, radar_data, bev_map_mask_g, bev_map_g, egocar_bev, \
            voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels = d

        # VoxelNet preprocessing
        voxel_input_feature_buffer = voxel_input_feature_buffer[:, 0]
        voxel_coordinate_buffer = voxel_coordinate_buffer[:, 0]
        number_of_occupied_voxels = number_of_occupied_voxels[:, 0]
        voxel_input_feature_buffer = voxel_input_feature_buffer.to(device)
        voxel_coordinate_buffer = voxel_coordinate_buffer.to(device)
        number_of_occupied_voxels = number_of_occupied_voxels.to(device)

    else:
        imgs, rots, trans, intrins, seg_bev_g, valid_bev_g, radar_data, bev_map_mask_g, bev_map_g, egocar_bev = d

    B0, T, S, C, H, W = imgs.shape
    assert (T == 1)

    # eliminate the time dimension
    imgs = imgs[:, 0]
    rots = rots[:, 0]
    trans = trans[:, 0]
    intrins = intrins[:, 0]  # intrinsics for each cam --> shape:  [B,S,4,4
    seg_bev_g = seg_bev_g[:, 0]
    valid_bev_g = valid_bev_g[:, 0]
    radar_data = radar_data[:, 0]
    # added bev_map_gt
    bev_map_mask_g = bev_map_mask_g[:, 0]
    if use_obj_layer_only_on_map:
        bev_map_mask_g = bev_map_mask_g[:, :-1]  # remove attached object class
    bev_map_g = bev_map_g[:, 0]
    # added egocar in bev plane
    egocar_bev = egocar_bev[:, 0]

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5  # go to -0.5, 0.5

    seg_bev_g = seg_bev_g.to(device)
    obj_seg_bev_e = torch.zeros_like(seg_bev_g)
    valid_bev_g = valid_bev_g.to(device)
    # added bev_map_gt
    bev_map_mask_g = bev_map_mask_g.to(device)
    bev_map_mask_e = torch.zeros_like(bev_map_mask_g)
    bev_map_g = bev_map_g.to(device)
    bev_map_e = torch.zeros_like(bev_map_g)
    # added egocar in bev plane
    egocar_bev = egocar_bev.to(device)

    # create ego car color plane
    ego_plane = torch.zeros_like(bev_map_g).to(device)
    ego_plane[:, [0, 2]] = 0.0
    ego_plane[:, 1] = 1.0
    # combine ego car and map
    ego_car_on_map_g = bev_map_g * (1 - egocar_bev) + ego_plane * egocar_bev

    # create other cars plane
    other_cars_plane = torch.zeros_like(bev_map_g).to(device)
    other_cars_plane[:, [0, 1]] = 0.0
    other_cars_plane[:, 2] = 1.0
    # combine ego car other cars and map
    ego_other_cars_on_map_g = ego_car_on_map_g * (1 - seg_bev_g) + other_cars_plane * seg_bev_g
    ego_other_cars_on_map_e = torch.zeros_like(ego_other_cars_on_map_g)

    rad_data = radar_data.to(device).permute(0, 2, 1)  # B, R, 19
    xyz_rad = rad_data[:, :, :3]
    meta_rad = rad_data[:, :, 3:]
    shallow_meta_rad = rad_data[:, :, 5:8]

    B, S, C, H, W = rgb_camXs.shape

    def __p(x):
        # Wrapper function: e.g. unites B,S dim to B*S
        return utils.basic.pack_seqdim(x, B)

    def __u(x):
        # Wrapper function: e.g. splits B*S dim into B,S
        return utils.basic.unpack_seqdim(x, B)

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))

    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_rad)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    if not model.module.use_radar:
        in_occ_mem0 = None
    elif model.module.use_radar and (model.module.use_metaradar or use_shallow_metadata):
        if use_radar_encoder and radar_encoder_type == 'voxel_net':
            voxelnet_feats_mem0 = voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels
            in_occ_mem0 = voxelnet_feats_mem0
        elif use_shallow_metadata:
            shallow_metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, shallow_meta_rad, Z, Y, X,
                                                                       assert_cube=False)
            in_occ_mem0 = shallow_metarad_occ_mem0
        else:  # use_metaradar
            metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)
            in_occ_mem0 = metarad_occ_mem0
    elif model.module.use_radar:
        rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
        in_occ_mem0 = rad_occ_mem0
    elif model.module.use_metaradar or use_shallow_metadata:
        assert False  # cannot use_metaradar without use_radar

    seg_e = model(
        rgb_camXs=rgb_camXs,
        pix_T_cams=pix_T_cams,
        cam0_T_camXs=cam0_T_camXs,
        vox_util=vox_util,
        rad_occ_mem0=in_occ_mem0)

    # get bev map from masks
    if train_task == 'both' or train_task == 'map':

        if train_task == 'both':
            bev_map_mask_e = seg_e[:, :-1]
            obj_seg_bev_e = seg_e[:, -1].unsqueeze(dim=1)
            obj_seg_bev = torch.sigmoid(obj_seg_bev_e)

            bev_map_only_mask_g = bev_map_mask_g
        else:
            bev_map_mask_e = seg_e
            obj_seg_bev = seg_bev_g  # add gt vehicles on map (optional)
            bev_map_only_mask_g = bev_map_mask_g

        map_seg_threshold = 0.4
        bev_map_e = nuscenes_data.get_rgba_map_from_mask2_on_batch(
            torch.sigmoid(bev_map_mask_e).detach().cpu().numpy(),
            threshold=map_seg_threshold, a=0.4).to(device)

        # combine ego car and bev_map_e
        ego_car_on_map_e = bev_map_e * (1 - egocar_bev) + ego_plane * egocar_bev

        # create other cars estimate plane
        other_cars_plane_e = torch.zeros_like(bev_map_e).to(device)
        other_cars_plane_e[:, [0, 1]] = 0.0
        other_cars_plane_e[:, 2] = 1.0

        # combine ego car other cars and map
        ego_other_cars_on_map_e = ego_car_on_map_e * (1 - obj_seg_bev) + other_cars_plane_e * obj_seg_bev

        # loss calculation
        map_seg_fc_loss = map_seg_loss_fn(bev_map_mask_e, bev_map_only_mask_g)
        #   map
        fc_map_factor = 1 / torch.exp(model.module.fc_map_weight)
        map_seg_fc_loss = 20.0 * map_seg_fc_loss * fc_map_factor  # 20.0
        # add to total loss
        total_loss += map_seg_fc_loss

        # MAP IoU calculation
        # ious for map segmentation:
        tp = ((torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & bev_map_mask_g.bool()).sum(dim=[2, 3])
        fp = ((torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & ~bev_map_mask_g.bool()).sum(dim=[2, 3])
        fn = (~(torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & bev_map_mask_g.bool()).sum(dim=[2, 3])

        map_intersections_per_class = tp.sum(dim=0)  # sum over batch --> 7 intersection values
        map_unions_per_class = (tp.sum(dim=0) + fp.sum(dim=0) + fn.sum(dim=0) + 1e-4)  # sum o. batch --> 7 union values

        # ################# NEW MULTI-IOU CALCULATION #####################
        map_seg_thresholds = torch.Tensor([0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]).to(device)
        sig_map_bev_e_new = torch.sigmoid(bev_map_mask_e)[:, :, :, :, None] >= map_seg_thresholds
        bev_map_mask_g_new = bev_map_only_mask_g[:, :, :, :, None]

        tps = (sig_map_bev_e_new.bool() & bev_map_mask_g_new.bool()).sum(dim=[2, 3])  # (B,7,12)
        fps = (sig_map_bev_e_new.bool() & ~bev_map_mask_g_new.bool()).sum(dim=[2, 3])
        fns = (~sig_map_bev_e_new.bool() & bev_map_mask_g_new.bool()).sum(dim=[2, 3])

        # best i/u
        map_masks_multi_ious_intersections = tps.sum(0)
        map_masks_multi_ious_unions = (tps.sum(0) + fps.sum(0) + fns.sum(0) + 1e-4)

        # metrics
        metrics['focal_loss_map'] = map_seg_fc_loss  # .item()
        metrics['fc_map_weight'] = model.module.fc_map_weight.item()
        # single threshold IoUs (t=0.4)
        metrics['map_masks_intersections'] = map_intersections_per_class
        metrics['map_masks_unions'] = map_unions_per_class
        # multi threshold IoUs
        metrics['map_masks_multi_ious_intersections'] = map_masks_multi_ious_intersections
        metrics['map_masks_multi_ious_unions'] = map_masks_multi_ious_unions
        metrics['map_seg_thresholds'] = map_seg_thresholds

    # object seg task
    if train_task == 'both' or train_task == 'object':
        if train_task == 'both':
            obj_seg_bev_e = seg_e[:, -1].unsqueeze(dim=1)
        else:  # 'object'
            obj_seg_bev_e = seg_e
            obj_seg_bev_e_sigmoid = torch.sigmoid(obj_seg_bev_e)
            ego_other_cars_on_map_e = ego_car_on_map_g * (1 - obj_seg_bev_e_sigmoid) + \
                other_cars_plane * obj_seg_bev_e_sigmoid
        # clc loss
        ce_loss = loss_fn(obj_seg_bev_e, seg_bev_g, valid_bev_g)
        # obj
        ce_factor = 1 / torch.exp(model.module.ce_weight)
        ce_loss = 10.0 * ce_loss * ce_factor  # 10.0
        total_loss += ce_loss

        # object IoUs
        obj_seg_bev_e_round = torch.sigmoid(obj_seg_bev_e).round()
        obj_intersection = (obj_seg_bev_e_round * seg_bev_g * valid_bev_g).sum(dim=[1, 2, 3])
        obj_union = ((obj_seg_bev_e_round + seg_bev_g) * valid_bev_g).clamp(0, 1).sum(dim=[1, 2, 3])

        obj_intersections = obj_intersection.sum()
        obj_unions = obj_union.sum()

        metrics['ce_loss'] = ce_loss
        metrics['ce_weight'] = model.module.ce_weight.item()
        metrics['obj_intersections'] = obj_intersections
        metrics['obj_unions'] = obj_unions

    if sw is not None and sw.save_this and is_master:
        if model.module.use_radar:
            rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)

            rad_occ_mem0_wandb = sw.summ_occ('0_inputs/rad_occ_mem0', rad_occ_mem0)
            rad_occ_mem0_wandb = rad_occ_mem0_wandb.squeeze().permute(1, 2, 0).numpy()
            rad_occ_mem0_wandb = wandb.Image(rad_occ_mem0_wandb)
            wandb.log({'train/inputs/rad_occ_mem0': rad_occ_mem0_wandb}, commit=False)

        rgb_input = sw.summ_rgb('0_inputs/rgb_camXs', torch.cat(rgb_camXs[0:1].unbind(1), dim=-1))  # 1,1,3,448,4800
        rgb_input = rgb_input.squeeze().permute(1, 2, 0).numpy()  # 448,4800,3
        rgb_input_wandb = wandb.Image(rgb_input)
        wandb.log({'train/inputs/rgb_camXs': rgb_input_wandb}, commit=False)

        ego_other_cars_on_map_g_img = ego_other_cars_on_map_g[0].permute(1, 2, 0).detach().cpu().numpy()
        ego_other_cars_on_map_g_wandb = wandb.Image(ego_other_cars_on_map_g_img)

        # own map estimate + object seg estimate + ego car
        ego_other_cars_on_map_e_img = ego_other_cars_on_map_e[0].permute(1, 2, 0).detach().cpu().numpy()
        ego_other_cars_on_map_e_wandb = wandb.Image(ego_other_cars_on_map_e_img)

        if train_task == 'both' or train_task == 'map':
            bev_map_mask_driv_g_img = bev_map_mask_g[0][0].detach().cpu().numpy()
            bev_map_mask_driv_g_img_wandb = wandb.Image(bev_map_mask_driv_g_img)

            bev_map_mask_park_g_img = bev_map_mask_g[0][1].detach().cpu().numpy()
            bev_map_mask_park_g_img_wandb = wandb.Image(bev_map_mask_park_g_img)

            bev_map_mask_ped_cross_g_img = bev_map_mask_g[0][2].detach().cpu().numpy()
            bev_map_mask_ped_cross_g_img_wandb = wandb.Image(bev_map_mask_ped_cross_g_img)

            bev_map_mask_walkway_g_img = bev_map_mask_g[0][3].detach().cpu().numpy()
            bev_map_mask_walkway_g_img_wandb = wandb.Image(bev_map_mask_walkway_g_img)

            bev_map_mask_stop_g_img = bev_map_mask_g[0][4].detach().cpu().numpy()
            bev_map_mask_stop_g_img_wandb = wandb.Image(bev_map_mask_stop_g_img)

            bev_map_mask_road_div_g_img = bev_map_mask_g[0][5].detach().cpu().numpy()
            bev_map_mask_road_div_g_img_wandb = wandb.Image(bev_map_mask_road_div_g_img)

            bev_map_mask_lane_div_g_img = bev_map_mask_g[0][6].detach().cpu().numpy()
            bev_map_mask_lane_div_g_img_wandb = wandb.Image(bev_map_mask_lane_div_g_img)

            bev_map_g_img = bev_map_g[0].permute(1, 2, 0).detach().cpu().numpy()
            bev_map_g_img_wandb = wandb.Image(bev_map_g_img)

            # ####################################       OUTPUTS     ###################################################
            # add e seg bev map image to wandb
            bev_map_e_img = bev_map_e[0].permute(1, 2, 0).detach().cpu().numpy()  # 200, 200, 3 don"t forget detach cpu
            bev_map_e_img_wandb = wandb.Image(bev_map_e_img)

            # we need the sigmoid outputs of the masks
            bev_map_mask_e_sig = torch.sigmoid(bev_map_mask_e)

            bev_map_mask_driv_e_img = bev_map_mask_e_sig[0][0].detach().cpu().numpy()
            bev_map_mask_driv_e_img_wandb = wandb.Image(bev_map_mask_driv_e_img)

            bev_map_mask_park_e_img = bev_map_mask_e_sig[0][1].detach().cpu().numpy()
            bev_map_mask_park_e_img_wandb = wandb.Image(bev_map_mask_park_e_img)

            bev_map_mask_ped_cross_e_img = bev_map_mask_e_sig[0][2].detach().cpu().numpy()
            bev_map_mask_ped_cross_e_img_wandb = wandb.Image(bev_map_mask_ped_cross_e_img)

            bev_map_mask_walkway_e_img = bev_map_mask_e_sig[0][3].detach().cpu().numpy()
            bev_map_mask_walkway_e_img_wandb = wandb.Image(bev_map_mask_walkway_e_img)

            bev_map_mask_stop_e_img = bev_map_mask_e_sig[0][4].detach().cpu().numpy()
            bev_map_mask_stop_e_img_wandb = wandb.Image(bev_map_mask_stop_e_img)

            bev_map_mask_road_div_e_img = bev_map_mask_e_sig[0][5].detach().cpu().numpy()
            bev_map_mask_road_div_e_img_wandb = wandb.Image(bev_map_mask_road_div_e_img)

            bev_map_mask_lane_div_e_img = bev_map_mask_e_sig[0][6].detach().cpu().numpy()
            bev_map_mask_lane_div_e_img_wandb = wandb.Image(bev_map_mask_lane_div_e_img)

            # prepare for logging...
            wandb.log({
                # combined map with ego car and other cars all estimates
                'train/vis/ego_cars_map_e': ego_other_cars_on_map_e_wandb,
                # combined map with ego car and other cars
                'train/vis/ego_cars_map_gt': ego_other_cars_on_map_g_wandb,

                # masks:
                'train/inputs/masks/bev_map_mask_driv_gt': bev_map_mask_driv_g_img_wandb,
                'train/inputs/masks/bev_map_mask_park_gt': bev_map_mask_park_g_img_wandb,
                'train/inputs/masks/bev_map_mask_ped_cross_gt': bev_map_mask_ped_cross_g_img_wandb,
                'train/inputs/masks/bev_map_mask_walkway_gt': bev_map_mask_walkway_g_img_wandb,
                'train/inputs/masks/bev_map_mask_stop_gt': bev_map_mask_stop_g_img_wandb,
                'train/inputs/masks/bev_map_mask_road_div_gt': bev_map_mask_road_div_g_img_wandb,
                'train/inputs/masks/bev_map_mask_lane_div_gt': bev_map_mask_lane_div_g_img_wandb,

                # ####### OUTPUTS ####### #
                # bev_map_estimate
                'train/outputs/bev_map_e': bev_map_e_img_wandb,
                'train/outputs/bev_map_g': bev_map_g_img_wandb,

                # masks
                'train/outputs/masks/bev_map_mask_driv_e': bev_map_mask_driv_e_img_wandb,
                'train/outputs/masks/bev_map_mask_park_e': bev_map_mask_park_e_img_wandb,
                'train/outputs/masks/bev_map_mask_ped_cross_e': bev_map_mask_ped_cross_e_img_wandb,
                'train/outputs/masks/bev_map_mask_walkway_e': bev_map_mask_walkway_e_img_wandb,
                'train/outputs/masks/bev_map_mask_stop_e': bev_map_mask_stop_e_img_wandb,
                'train/outputs/masks/bev_map_mask_road_div_e': bev_map_mask_road_div_e_img_wandb,
                'train/outputs/masks/bev_map_mask_lane_div_e': bev_map_mask_lane_div_e_img_wandb,

            }, commit=False)

        if train_task == 'both' or train_task == 'object':
            seg_bev_g_wandb = sw.summ_oned('2_outputs/obj_seg_bev_g', seg_bev_g * (0.5 + valid_bev_g * 0.5),
                                           norm=False)  # 1,1,3,200,200
            seg_bev_g_wandb = seg_bev_g_wandb.squeeze().permute(1, 2, 0).numpy()
            seg_bev_g_wandb = wandb.Image(seg_bev_g_wandb)

            valid_bev_g_wandb = sw.summ_oned('2_outputs/valid_obj_bev_g', valid_bev_g, norm=False)  # 1,1,3,200,200
            valid_bev_g_wandb = valid_bev_g_wandb.squeeze().permute(1, 2, 0).numpy()
            valid_bev_g_wandb = wandb.Image(valid_bev_g_wandb)

            seg_bev_e_wandb = sw.summ_oned('2_outputs/obj_seg_bev_e', torch.sigmoid(obj_seg_bev_e).round(),
                                           norm=False)  # 1,1,3,200,200
            seg_bev_e_wandb = seg_bev_e_wandb.squeeze().permute(1, 2, 0).numpy()
            seg_bev_e_wandb = wandb.Image(seg_bev_e_wandb)

            seg_bev_e_soft_wandb = sw.summ_oned('2_outputs/obj_seg_bev_e_soft', torch.sigmoid(obj_seg_bev_e),
                                                norm=False)  # 1,1,3,200,200
            seg_bev_e_soft_wandb = seg_bev_e_soft_wandb.squeeze().permute(1, 2, 0).numpy()
            seg_bev_e_soft_wandb = wandb.Image(seg_bev_e_soft_wandb)

            wandb.log({
                # object seg GT
                'train/outputs/seg_bev_g': seg_bev_g_wandb,
                # valid GT seg mask
                'train/outputs/valid_bev_g': valid_bev_g_wandb,
                # object seg estimate
                'train/outputs/obj_seg_bev_e': seg_bev_e_wandb,
                # object seg estimate (smoothed)
                'train/outputs/obj_seg_bev_e_soft': seg_bev_e_soft_wandb,
            }, commit=False)

        print("FINISHED IMAGE LOGGING")

    return total_loss, metrics


def main(
        exp_name='bevcar_debug_ddp',
        # training
        max_iters=75000,
        log_freq=1000,
        shuffle=True,
        dset='trainval',
        save_freq=1000,
        batch_size=8,
        grad_acc=5,
        lr=3e-4,
        use_scheduler=True,
        weight_decay=1e-7,
        nworkers=12,
        # data/log/save/load directories
        data_dir='../nuscenes/',
        custom_dataroot='../../custom_nuscenes/scaled_images',
        log_dir='logs_nuscenes_bevcar',
        ckpt_dir='checkpoints/',
        keep_latest=75,
        init_dir='',
        ignore_load=None,
        load_step=False,
        load_optimizer=False,
        load_scheduler=False,
        # data
        final_dim=[448, 896],  # to match //8, //14, //16 and //32 in Vit
        rand_flip=True,
        rand_crop_and_resize=True,
        ncams=6,
        nsweeps=5,
        # model
        encoder_type='dino_v2',
        radar_encoder_type='voxel_net',
        use_rpn_radar=False,
        train_task='both',
        use_radar=False,
        use_radar_filters=False,
        use_radar_encoder=False,
        use_metaradar=False,
        use_shallow_metadata=False,
        use_pre_scaled_imgs=False,
        use_obj_layer_only_on_map=False,
        init_query_with_image_feats=False,
        do_rgbcompress=True,
        do_shuffle_cams=True,
        use_multi_scale_img_feats=False,
        num_layers=6,
        # cuda
        device_ids=None,
        freeze_dino=True,
        do_feat_enc_dec=True,
        combine_feat_init_w_learned_q=True,
        model_type='transformer',
        use_radar_occupancy_map=False,
        learnable_fuse_query=True,
        # wandb
        group='debug',
        notes='debug run',
        name='debug',
):
    assert (model_type in ['transformer', 'simple_lift_fuse', 'SimpleBEV_map'])
    # debug only  - check for CUDA
    if device_ids is None:
        device_ids = [0]

    B = batch_size

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default
    # Distributed process initialization
    is_master = False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if rank == 0:
        is_master = True

    torch.distributed.init_process_group(backend='nccl', init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)  # to avoid "invalid device ordinal" error

    # print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)

    if grad_acc > 1 and is_master:
        print('effective batch size:', B * grad_acc * world_size)

    torch.cuda.manual_seed(125)
    torch.cuda.synchronize()
    # needed referring to:
    # https://towardsdatascience.com/distribute-your-pytorch-model-in-less-than-20-lines-of-code-61a786e6e7b0
    # only if problem with GPU memory:
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{local_rank}")

    dist.barrier()
    # debug only
    if torch.cuda.is_available():
        if is_master:
            print("CUDA is available")
            print("Devices available: %d " % torch.cuda.device_count())
            print("############### GPU CACHE EMPTIED ###############")
        dist.barrier()
        if is_master:  # split only for sequential printing
            print(f"\nMASTER process: rank {rank}, local_rank {local_rank}", flush=True)
        else:
            print(f"\nWORKER process: rank {rank}, local_rank {local_rank}", flush=True)
    else:
        print("CUDA is --- NOT --- available")
    dist.barrier()

    # autogen a name
    model_name = "%d" % B
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    if world_size > 1:
        model_name += "x%d" % world_size
    lrn = "%.1e" % lr  # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"

    import datetime
    model_date = datetime.datetime.now().strftime('%H-%M-%S')
    model_name = model_name + '_' + model_date

    model_name = exp_name + '_' + model_name

    # set up ckpt and logging
    ckpt_dir = os.path.join(ckpt_dir, model_name)

    dist.barrier()
    if rank == 0:
        try:
            writer_t = SummaryWriter(os.path.join(log_dir, model_name + '/t'), max_queue=10, flush_secs=60)
        except FileExistsError:
            pass
    else:
        writer_t = None

    dist.barrier()

    if is_master:
        print('model_name', model_name)
        print('resolution:', final_dim)

        # print radar encoder
        if use_radar_encoder:
            print("Radar encoder: ", radar_encoder_type)
        else:
            print("NO RADAR ENCODER")

    # wandb extension
    wandb_config = {
        "exp_name": exp_name,
        "final_dim": final_dim,
        # training
        "max_iters": max_iters,
        "log_freq": log_freq,
        "shuffle": shuffle,
        "dset": dset,
        "save_freq": save_freq,
        "batch_size": batch_size,
        "grad_acc": grad_acc,
        "lr": lr,
        "use_scheduler": use_scheduler,
        "weight_decay": weight_decay,
        "nworkers": nworkers,
        # data/log/save/load directories
        "data_dir": data_dir,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "keep_latest": keep_latest,
        "init_dir": init_dir,
        "ignore_load": ignore_load,
        "load_step": load_step,
        "load_optimizer": load_optimizer,
        "load_scheduler": load_scheduler,
        # data
        "rand_flip": rand_flip,
        "rand_crop_and_resize": rand_crop_and_resize,
        "ncams": ncams,
        "nsweeps": nsweeps,
        # model
        "encoder_type": encoder_type,
        "radar_encoder_type": radar_encoder_type,
        "use_radar": use_radar,
        "use_radar_filters": use_radar_filters,
        "use_metaradar": use_metaradar,
        "use_shallow_metadata": use_shallow_metadata,
        "do_rgbcompress": do_rgbcompress,
        "do_shuffle_cams": do_shuffle_cams,
        # cuda
        "device_ids": device_ids,
        "use_pre_scaled_imgs": use_pre_scaled_imgs,
        "use_obj_layer_only_on_map": use_obj_layer_only_on_map,
        "init_query_with_image_feats": init_query_with_image_feats,
        "use_multi_scale_img_feats": use_shallow_metadata,
        "num_layers": num_layers,
        "freeze_dino": freeze_dino,
        "model_type": model_type,
        "use_radar_occupancy_map": use_radar_occupancy_map,
        "combine_feat_init_w_learned_q": combine_feat_init_w_learned_q,
    }

    wandb.init(project=model_name, config=wandb_config, group=group, notes=notes, name=name)
    dist.barrier()

    if rand_crop_and_resize:
        resize_lim = [0.8, 1.2]
        crop_offset = int(final_dim[0] * (1 - resize_lim[0]))
    else:
        resize_lim = [1.0, 1.0]
        crop_offset = 0

    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }
    train_dataloader, _ = nuscenes_data.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z, Y, X),
        bsz=B,
        nworkers=nworkers,
        shuffle=shuffle,
        use_radar_filters=use_radar_filters,
        seqlen=1,
        nsweeps=nsweeps,
        do_shuffle_cams=do_shuffle_cams,
        distributed_sampler=True,
        rank=rank,
        radar_encoder_type=radar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        use_radar_occupancy_map=use_radar_occupancy_map,
    )
    train_iterloader = iter(train_dataloader)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    # set up model & losses
    if is_master:
        print("MODEL TYPE: " + model_type)

    # Transformer based lifting and fusion -> BEVCar
    if model_type == 'transformer':
        model = SegnetTransformerLiftFuse(Z_cam=200, Y_cam=8, X_cam=200, Z_rad=Z, Y_rad=Y, X_rad=X, vox_util=None,
                                          use_radar=use_radar, use_metaradar=use_metaradar,
                                          use_shallow_metadata=use_shallow_metadata,
                                          use_radar_encoder=use_radar_encoder, do_rgbcompress=do_rgbcompress,
                                          encoder_type=encoder_type, radar_encoder_type=radar_encoder_type,
                                          rand_flip=rand_flip, train_task=train_task,
                                          init_query_with_image_feats=init_query_with_image_feats,
                                          use_obj_layer_only_on_map=use_obj_layer_only_on_map,
                                          do_feat_enc_dec=do_feat_enc_dec,
                                          use_multi_scale_img_feats=use_multi_scale_img_feats, num_layers=num_layers,
                                          latent_dim=128,
                                          combine_feat_init_w_learned_q=combine_feat_init_w_learned_q,
                                          use_rpn_radar=use_rpn_radar, use_radar_occupancy_map=use_radar_occupancy_map,
                                          freeze_dino=freeze_dino, learnable_fuse_query=learnable_fuse_query,
                                          is_master=is_master)

    elif model_type == 'simple_lift_fuse':
        # our net with replaced lifting and fusion from SimpleBEV
        model = SegnetSimpleLiftFuse(Z_cam=200, Y_cam=8, X_cam=200, Z_rad=Z, Y_rad=Y, X_rad=X, vox_util=None,
                                     use_radar=use_radar, use_metaradar=use_metaradar,
                                     use_shallow_metadata=use_shallow_metadata, use_radar_encoder=use_radar_encoder,
                                     do_rgbcompress=do_rgbcompress, encoder_type=encoder_type,
                                     radar_encoder_type=radar_encoder_type, rand_flip=rand_flip, train_task=train_task,
                                     use_obj_layer_only_on_map=use_obj_layer_only_on_map,
                                     do_feat_enc_dec=do_feat_enc_dec,
                                     use_multi_scale_img_feats=use_multi_scale_img_feats, num_layers=num_layers,
                                     latent_dim=128, use_rpn_radar=use_rpn_radar,
                                     use_radar_occupancy_map=use_radar_occupancy_map,
                                     freeze_dino=freeze_dino, is_master=is_master)

    else:  # model_type == 'SimpleBEV_map':
        model = SegnetWithMap(Z, Y, X, vox_util=vox_util, use_radar=use_radar,
                              use_metaradar=use_metaradar, use_shallow_metadata=use_shallow_metadata,
                              do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=rand_flip,
                              train_task=train_task, freeze_dino=freeze_dino, is_master=is_master)

    # DDP adaptation of BN layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    parameters = list(model.parameters())
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, model.parameters())
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        scheduler = None

    if is_master:
        # Counting trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')
        # Counting non-trainable parameters
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f'Non-trainable parameters: {non_trainable_params}')
        # Overall parameters
        total_params = trainable_params + non_trainable_params
        print('Total parameters (trainable + fixed)', total_params)

    # load checkpoint
    global_step = 0
    if init_dir:
        if load_step and load_optimizer and load_scheduler:
            global_step = saverloader.load(init_dir, model, optimizer, scheduler=scheduler,
                                           ignore_load=ignore_load, device_ids=device, is_DP=False)
        elif load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model, optimizer, ignore_load=ignore_load,
                                           device_ids=device)
            print("global_step: ", global_step)
        elif load_step:
            global_step = saverloader.load(init_dir, model, ignore_load=ignore_load, device_ids=device)
        else:
            _ = saverloader.load(init_dir, model, ignore_load=ignore_load)
            global_step = 0

        optimizer.param_groups[0]['capturable'] = True

    model = DDP(model, device_ids=[local_rank], static_graph=True)

    # for obj. segmentation head
    seg_loss_fn = SimpleLoss(2.13).to(device)  # value from lift-splat
    # for map segmentation head
    map_seg_loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=3, reduction="sum_of_class_means").to(device)

    # wandb setup watcher
    wandb.watch(model, log_freq=log_freq)
    model.train()

    # set up running logging pools
    n_pool_train = 10
    train_pool_dict, train_pool_dict_name = create_train_pool_dict(name='t', n_pool=n_pool_train)
    ddp_train_pool_dict, ddp_train_pool_dict_name = create_train_pool_dict(name='t_ddp', n_pool=n_pool_train)

    sw_t = None

    # training loop
    while global_step < max_iters:
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0

        metrics = {}
        metrics_mean_grad_acc = {}
        total_loss = 0.0

        for internal_step in range(grad_acc):
            # read sample
            read_start_time = time.time()

            if internal_step == grad_acc - 1:
                sw_t = utils.improc.Summ_writer(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=2,
                    scalar_freq=int(log_freq / 2),
                    just_gif=True)
            else:
                sw_t = None

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)

            read_time = time.time() - read_start_time
            iter_read_time += read_time

            # run training iteration
            total_loss_, metrics_ = run_model(model, seg_loss_fn,
                                              map_seg_loss_fn,
                                              sample, device, sw_t, use_radar_encoder, radar_encoder_type, train_task,
                                              is_master=is_master, use_shallow_metadata=use_shallow_metadata,
                                              use_obj_layer_only_on_map=use_obj_layer_only_on_map)

            (total_loss_ / grad_acc).backward()

            # collect total loss and metrics over grad_acc steps
            if internal_step == 0:
                metrics_mean_grad_acc = metrics_
                total_loss = total_loss_
            else:
                # accumulates metrics and if end of grad_acc -> divides by grad_acc --> except intersections / unions
                metrics = grad_acc_metrics(metrics_single_pass=metrics_, metrics_mean_grad_acc=metrics_mean_grad_acc,
                                           internal_step=internal_step,
                                           grad_acc=grad_acc)
                total_loss += total_loss_

            if internal_step == grad_acc - 1:
                total_loss = total_loss / grad_acc

        torch.nn.utils.clip_grad_norm_(parameters, 5.0)

        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        t_loss_reduced, metrics_reduced = reduce_loss_metrics(total_loss, metrics, train_task=train_task)
        t_train_loss = total_loss
        t_train_loss_reduced = t_loss_reduced
        collect_metrics_for_wandb(total_loss=t_loss_reduced, metrics=metrics_reduced, mode='train_ddp',
                                  pool_dict=ddp_train_pool_dict, pool_name=ddp_train_pool_dict_name,
                                  commit=False, train_task=train_task)

        # save model checkpoint
        if np.mod(global_step, save_freq) == 0 and rank == 0:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, scheduler=scheduler,
                             keep_latest=keep_latest)

        # wait until model is saved
        dist.barrier()

        model.train()

        # log lr and time
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            sw_t.summ_scalar('_/current_lr', current_lr)

        iter_time = time.time() - iter_start_time
        train_pool_dict['time_pool_' + train_pool_dict_name].update([iter_time])

        # wandb
        if global_step % log_freq == 0:
            wandb.log({
                'train': {
                    'pooled/time_per_batch': train_pool_dict['time_pool_' + train_pool_dict_name].mean(),
                    'pooled/time_per_el': train_pool_dict['time_pool_' + train_pool_dict_name].mean() / float(B),
                }
            }, commit=False)

        wandb.log({
            'train': {
                'params/current_lr': current_lr,
            }
        }, commit=True)

        if rank == 0:
            if train_task == 'both':
                print('%s; step %06d/%d; rtime %.2f; itime %.2f; rank_loss %.5f; DDP_loss %.5f; obj_iou_t %.1f; '
                      ' map_iou_t %.1f;' % (
                          model_name, global_step, max_iters, iter_read_time, iter_time,
                          t_train_loss, t_train_loss_reduced,
                          100 * ddp_train_pool_dict['obj_iou_pool_' + ddp_train_pool_dict_name].mean(),
                          100 * ddp_train_pool_dict['masks_mean_iou_pool_' + ddp_train_pool_dict_name].mean()))

            if train_task == 'object':
                print('%s; step %06d/%d; rtime %.2f; itime %.2f; rank_loss %.5f; DDP_loss %.5f; obj_iou_t %.1f; ' % (
                    model_name, global_step, max_iters, iter_read_time, iter_time,
                    t_train_loss, t_train_loss_reduced,
                    100 * ddp_train_pool_dict['obj_iou_pool_' + ddp_train_pool_dict_name].mean()))

            if train_task == 'map':
                print('%s; step %06d/%d; rtime %.2f; itime %.2f; rank_loss %.5f; DDP_loss %.5f; '
                      'map_iou_t %.1f;' % (
                          model_name, global_step, max_iters, iter_read_time, iter_time,
                          t_train_loss, t_train_loss_reduced,
                          100 * ddp_train_pool_dict['masks_mean_iou_pool_' + ddp_train_pool_dict_name].mean()))

    writer_t.close()
    wandb.finish()

    # cleanup the process group
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training with model-specific config.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(**config)
