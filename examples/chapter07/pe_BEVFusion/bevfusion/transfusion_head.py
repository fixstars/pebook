# Copyright (c) Fixstars. All rights reserved.
# modify from https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/projects/BEVFusion/bevfusion/transfusion_head.py
# modify from https://github.com/mit-han-lab/bevfusion

import copy
from typing import List, Tuple

import numba
import numpy as np
import nvtx
import torch
from mmdet.models.task_modules import AssignResult
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdetection3d.projects.BEVFusion.bevfusion.transfusion_head import TransFusionHead
from .gaussian import  draw_heatmap_gaussian, gaussian_radius

@MODELS.register_module()
class PE_TransFusionHead(TransFusionHead):
    @nvtx.annotate("get_targets", color="blue")
    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        return super().get_targets(batch_gt_instances_3d, preds_dict)

    @nvtx.annotate("get_targets_single", color="red")
    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]
        # PATCH BEGIN
        target_device = preds_dict['center'].device
        # PATCH END

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        with nvtx.annotate("bbox_coder.decode", color="blue"):
            boxes_dict = self.bbox_coder.decode(
                score, rot, dim, center, height,
                vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            with nvtx.annotate("idx_layer", color="blue"):
                bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                    idx_layer:self.num_proposals *
                                                    (idx_layer + 1), :]
                score_layer = score[..., self.num_proposals *
                                    idx_layer:self.num_proposals *
                                    (idx_layer + 1), ]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                with nvtx.annotate("HungarianAssigner3D", color="blue"):
                    assign_result = self.bbox_assigner.assign(
                        bboxes_tensor_layer,
                        gt_bboxes_tensor,
                        gt_labels_3d,
                        score_layer,
                        self.train_cfg,
                    )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # PATCH BEGIN: use cpu
        gt_bboxes_tensor = gt_bboxes_tensor.cpu()
        bboxes_tensor = bboxes_tensor.cpu()
        gt_labels_3d = gt_labels_3d.cpu()

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]).cpu(),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]).cpu(),
            labels=torch.cat([res.labels for res in assign_result_list]).cpu(),
        )
        # PATCH END

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        with nvtx.annotate("bbox_sampler.sample", color="blue"):
            sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                    pred_instances,
                                                    gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        # PATCH BEGIN: use cpu
        with nvtx.annotate("bbox_targets", color="blue"):
            bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                        ]).to(bboxes_tensor.device)
            bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                        ]).to(bboxes_tensor.device)
            ious = assign_result_ensemble.max_overlaps
            ious = torch.clamp(ious, min=0.0, max=1.0)
            labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
            label_weights = bboxes_tensor.new_zeros(
                num_proposals, dtype=torch.long)
        # PATCH END

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            with nvtx.annotate("bbox_coder.encode", color="blue"):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        # PATCH BEGIN: use draw_heatmaps
        with nvtx.annotate("heatmap targets", color="blue"):
            gt_bboxes_3d: np.ndarray = torch.cat(
                [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
                dim=1).detach().cpu().numpy()
            grid_size = np.array(self.train_cfg['grid_size'])
            pc_range = self.train_cfg['point_cloud_range']
            voxel_size = self.train_cfg['voxel_size']
            feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                                )  # [x_len, y_len]
            heatmap = np.zeros(shape=(self.num_classes, feature_map_size[1], feature_map_size[0]), dtype=gt_bboxes_3d.dtype)
            gt_labels_3d = gt_labels_3d.detach().cpu().numpy()

        draw_heatmaps(
            heatmap,
            gt_bboxes_3d,
            gt_labels_3d,
            (voxel_size[0], voxel_size[1]),
            self.train_cfg['out_size_factor'],
            self.train_cfg['gaussian_overlap'],
            self.train_cfg['min_radius'],
            (pc_range[0], pc_range[1]),
        )
        # PATCH END

        # PATCH BEGIN
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        heatmap = torch.from_numpy(heatmap).to(target_device, non_blocking=True)
        labels = labels.to(target_device, non_blocking=True)
        label_weights = label_weights.to(target_device, non_blocking=True)
        bbox_targets = bbox_targets.to(target_device, non_blocking=True)
        bbox_weights = bbox_weights.to(target_device, non_blocking=True)
        ious = ious.to(target_device, non_blocking=True)
        # PATCH END

        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    @nvtx.annotate("loss", color="blue")
    def loss(self, batch_feats, batch_data_samples):
        return super().loss(batch_feats, batch_data_samples)

    @nvtx.annotate("loss_by_feat", color="yellow")
    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        return super().loss_by_feat(preds_dicts, batch_gt_instances_3d, *args, **kwargs)

@numba.njit(cache=True, parallel=True)
def draw_heatmaps(heatmap: np.ndarray, gt_bboxes_3d: np.ndarray, gt_labels_3d: np.ndarray, voxel_size: Tuple[float, float], out_size_factor: float, gaussian_overlap: float, min_radius: int, pc_range: Tuple[float, float]):
    gaussian_input = [(np.nan, np.nan, 0) for _ in range(len(gt_bboxes_3d))]
    for idx in numba.prange(len(gt_bboxes_3d)):
        width = gt_bboxes_3d[idx][3].item()
        length = gt_bboxes_3d[idx][4].item()
        width = width / voxel_size[0] / out_size_factor
        length = length / voxel_size[1] / out_size_factor
        if width > 0 and length > 0:
            radius = gaussian_radius((length, width), min_overlap=gaussian_overlap)
            radius = max(min_radius, int(radius))
            x, y = gt_bboxes_3d[idx][0].item(), gt_bboxes_3d[idx][1].item()

            coor_x = ((x - pc_range[0]) / voxel_size[0] /
                        out_size_factor)
            coor_y = ((y - pc_range[1]) / voxel_size[1] /
                        out_size_factor)
            gaussian_input[idx] = (coor_x, coor_y, radius)
    for idx in range(len(gt_bboxes_3d)):
        if not np.isnan(gaussian_input[idx][0]):
            coor_x, coor_y, radius = gaussian_input[idx]
            # original
            # center_int = np.array([coor_x, coor_y], dtype=np.int32)
            # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
            # NOTE: fix
            center_int = np.array([coor_y, coor_x], dtype=np.int32)
            draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
