import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmdet3d.ops import scatter_v2
import torch_scatter
from mmdet.models.builder import build_loss

@HEADS.register_module()
class PanoSegOccHead(DETRHead):
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 bev_z=5,
                 voxel_lidar = [0.05, 0.05, 0.05],
                 voxel_det = [2.048,2.048,1],
                 loss_occupancy=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                loss_occupancy_aux = None,
                loss_occupancy_det=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                bg_weight = 0.02,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.voxel_lidar = voxel_lidar
        self.voxel_det = voxel_det
        self.fp16_enabled = False
        self.bg_weight = bg_weight

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(PanoSegOccHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.lidar_seg_loss = build_loss(loss_occupancy)
        self.lidar_det_loss = build_loss(loss_occupancy_det)
        if loss_occupancy_aux is not None:
            self.lidar_seg_aux_loss = build_loss(loss_occupancy_aux)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w, self.bev_z),device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            bev_feat, bev_embed, hs, init_reference, inter_references, occupancy, occupancy_det = outputs
            return bev_feat, bev_embed
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_feat, bev_embed, hs, init_reference, inter_references, occupancy, occupancy_det = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_feat': bev_feat,
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'occupancy': occupancy,
            'occupancy_det':occupancy_det,
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,:10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
    def get_occupancy_det_label(self,voxel_coors_det, voxel_label_det, occupancy_det_label):

        voxel_coors_det[:,1] = voxel_coors_det[:,1].clip(min=0,max=self.bev_z-1)
        voxel_coors_det[:,2] = voxel_coors_det[:,2].clip(min=0,max=self.bev_h-1)
        voxel_coors_det[:,3] = voxel_coors_det[:,3].clip(min=0,max=self.bev_w-1)

        det_label_binary = ((voxel_label_det>=1)&(voxel_label_det<=10))
        det_label = det_label_binary.long()
        occupancy_det_label[0,voxel_coors_det[:,1],voxel_coors_det[:,2],voxel_coors_det[:,3]]=det_label
        return occupancy_det_label
    
    def get_det_loss(self,voxel_label_det,occupancy_det_label,occupancy_det_pred):

        num_total_pos_det = len(voxel_label_det)

        num_total_neg_det = len(occupancy_det_label) - num_total_pos_det
        avg_factor_det = num_total_pos_det * 1.0 + num_total_neg_det * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor_det = reduce_mean(
                occupancy_det_pred.new_tensor([avg_factor_det]))
        avg_factor_det = max(avg_factor_det, 1)

        losses_det = self.lidar_det_loss(occupancy_det_pred,occupancy_det_label,avg_factor=avg_factor_det)
        return losses_det
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             pts_sem,
             preds_dicts,
             dense_occupancy = None,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

       # Extract the first three columns from pts_sem
        pts = pts_sem[:, :3]

        # Extract the fourth column from pts_sem
        pts_semantic_mask = pts_sem[:, 3:4]

        # If dense_occupancy is None, perform voxelization and label voxelization
        if dense_occupancy is None:
            pts_coors, voxelized_data, voxel_coors = self.voxelize(pts, self.pc_range, self.voxel_lidar)
            voxel_label = self.label_voxelization(pts_semantic_mask, pts_coors, voxel_coors)

        # Perform voxelization and label voxelization for detection
        pts_coors_det, voxelized_data_det, voxel_coors_det = self.voxelize(pts, self.pc_range, self.voxel_det)
        voxel_label_det = self.label_voxelization(pts_semantic_mask, pts_coors_det, voxel_coors_det)
        
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        occupancy = preds_dicts['occupancy']
        occupancy_det = preds_dicts['occupancy_det']

        occupancy_pred = occupancy.squeeze(0)
        occupancy_det_pred = occupancy_det[0].squeeze(0)

        cls_num,occ_z,occ_h,occ_w = occupancy_pred.shape
        if dense_occupancy is None:
            occupancy_label = torch.full((1, occ_z, occ_h, occ_w), cls_num, device=occupancy_pred.device, dtype=torch.long)
        else:
            occupancy_label = (torch.zeros(1,occ_z,occ_h,occ_w)).to(occupancy_pred.device).long()
       
        occupancy_det_label = (torch.ones(1,self.bev_z,self.bev_h,self.bev_w)*2).to(occupancy_det_pred.device).long()

        if dense_occupancy is None:
            voxel_coors[:,1] = voxel_coors[:,1].clip(min=0,max=occ_z-1)
            voxel_coors[:,2] = voxel_coors[:,2].clip(min=0,max=occ_h-1)
            voxel_coors[:,3] = voxel_coors[:,3].clip(min=0,max=occ_w-1)
            occupancy_label[0,voxel_coors[:,1],voxel_coors[:,2],voxel_coors[:,3]] = voxel_label
        else:
            dense_occupancy = dense_occupancy.long().squeeze(0)
            occupancy_label[0,dense_occupancy[:,0],dense_occupancy[:,1],dense_occupancy[:,2]]=dense_occupancy[:,3]

        occupancy_det_label = self.get_occupancy_det_label(voxel_coors_det, voxel_label_det, occupancy_det_label)

        losses_seg_aux = self.lidar_seg_aux_loss(occupancy_pred.unsqueeze(0),occupancy_label)

        occupancy_det_label = occupancy_det_label.reshape(-1)
        occupancy_label = occupancy_label.reshape(-1)

        assert occupancy_label.max()<=cls_num and occupancy_label.min()>=0
        occupancy_pred = occupancy_pred.reshape(cls_num,-1).permute(1,0)
        occupancy_det_pred = occupancy_det_pred.reshape(2,-1).permute(1,0)

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # Lidar seg loss
        if dense_occupancy is None:
            num_total_pos = len(voxel_label)
        else:
            num_total_pos = len(dense_occupancy)
        num_total_neg = len(occupancy_label)-num_total_pos
        avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(
                occupancy_pred.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        losses_seg = self.lidar_seg_loss(occupancy_pred,occupancy_label,avg_factor=avg_factor)

        # Lidar det loss
        losses_det = self.get_det_loss(voxel_label_det,occupancy_det_label,occupancy_det_pred)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_seg'] = losses_seg
        loss_dict['loss_det'] = losses_det
        loss_dict['loss_seg_aux'] = losses_seg_aux

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss_new(self,
             gt_bboxes_list,
             gt_labels_list,
             pts_sem,
             preds_dicts,
             dense_occupancy = None,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # GT voxel supervision
        pts = pts_sem[:,:3]
        pts_semantic_mask = pts_sem[:,3:4]

        pts_numpy = pts.cpu().numpy()
        pts_semantic_mask_numpy = pts_semantic_mask.cpu().numpy()
        points_grid_ind = np.floor((np.clip(pts_numpy, self.pc_range[:3],self.pc_range[3:]) - self.pc_range[:3]) / self.voxel_lidar).astype(np.int)
        label_voxel_pair = np.concatenate([points_grid_ind, pts_semantic_mask_numpy], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((points_grid_ind[:, 0], points_grid_ind[:, 1], points_grid_ind[:, 2])), :]
        label_voxel = torch.tensor(label_voxel_pair).to(pts.device).long()
        if dense_occupancy is None:
            pts_coors,voxelized_data,voxel_coors = self.voxelize(pts,self.pc_range,self.voxel_lidar)
            voxel_label = self.label_voxelization(pts_semantic_mask, pts_coors, voxel_coors)

        pts_coors_det,voxelized_data_det,voxel_coors_det = self.voxelize(pts,self.pc_range,self.voxel_det)
        voxel_label_det = self.label_voxelization(pts_semantic_mask, pts_coors_det, voxel_coors_det)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        occupancy = preds_dicts['occupancy']
        occupancy_det = preds_dicts['occupancy_det']

        occupancy_pred = occupancy.squeeze(0)
        occupancy_det_pred = occupancy_det.squeeze(0)

        cls_num,occ_z,occ_h,occ_w = occupancy_pred.shape
        if dense_occupancy is None:
            occupancy_label = (torch.ones(1,occ_z,occ_h,occ_w)*cls_num).to(occupancy_pred.device).long()
        else:
            occupancy_label = (torch.zeros(1,occ_z,occ_h,occ_w)).to(occupancy_pred.device).long()
        occupancy_det_label = (torch.ones(1,self.bev_z,self.bev_h,self.bev_w)*2).to(occupancy_det_pred.device).long()

        # Matrix operation acceleration
        if dense_occupancy is None:
            occupancy_label[0,label_voxel[:,2],label_voxel[:,1],label_voxel[:,0]] = label_voxel[:,3]
        else:
            dense_occupancy = dense_occupancy.long().squeeze(0)
            occupancy_label[0,dense_occupancy[:,0],dense_occupancy[:,1],dense_occupancy[:,2]]=dense_occupancy[:,3]

        voxel_coors_det[:,1] = voxel_coors_det[:,1].clip(min=0,max=self.bev_z-1)
        voxel_coors_det[:,2] = voxel_coors_det[:,2].clip(min=0,max=self.bev_h-1)
        voxel_coors_det[:,3] = voxel_coors_det[:,3].clip(min=0,max=self.bev_w-1)

        det_label_binary = ((voxel_label_det>=1)&(voxel_label_det<=10))
        det_label = det_label_binary.long()
        occupancy_det_label[0,voxel_coors_det[:,1],voxel_coors_det[:,2],voxel_coors_det[:,3]]=det_label

        losses_seg_aux = self.lidar_seg_aux_loss(occupancy_pred.unsqueeze(0),occupancy_label)

        occupancy_det_label = occupancy_det_label.reshape(-1)
        occupancy_label = occupancy_label.reshape(-1)


        assert occupancy_label.max()<=cls_num and occupancy_label.min()>=0
        occupancy_pred = occupancy_pred.reshape(cls_num,-1).permute(1,0)
        occupancy_det_pred = occupancy_det_pred.reshape(2,-1).permute(1,0)

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # Lidar seg loss
        if dense_occupancy is None:
            num_total_pos = len(voxel_label)
        else:
            num_total_pos = len(dense_occupancy)
        num_total_neg = len(occupancy_label)-num_total_pos
        avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(
                occupancy_pred.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        losses_seg = self.lidar_seg_loss(occupancy_pred,occupancy_label,avg_factor=avg_factor)

        # Lidar det loss
        num_total_pos_det = len(voxel_label_det)


        num_total_neg_det = len(occupancy_det_label)-num_total_pos_det
        avg_factor_det = num_total_pos_det * 1.0 + num_total_neg_det * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor_det = reduce_mean(
                occupancy_det_pred.new_tensor([avg_factor_det]))
        avg_factor_det = max(avg_factor_det, 1)

        losses_det = self.lidar_det_loss(occupancy_det_pred,occupancy_det_label,avg_factor=avg_factor_det)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_seg'] = losses_seg
        loss_dict['loss_det'] = losses_det
        loss_dict['loss_seg_aux'] = losses_seg_aux

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list

    def decode_lidar_seg(self,points,occupancy):

        pts_coors,voxelized_data,voxel_coors = self.voxelize(points,self.pc_range,self.voxel_lidar)

        # clip out-ranged points
        z_max = int((self.pc_range[5]-self.pc_range[2])/self.voxel_lidar[2])-1
        y_max = int((self.pc_range[4]-self.pc_range[1])/self.voxel_lidar[1])-1
        x_max = int((self.pc_range[3]-self.pc_range[0])/self.voxel_lidar[0])-1

        # valid_mask = (pts_coors[:,1].cpu().numpy()>=0) & (pts_coors[:,1].cpu().numpy()<=z_max) \
        #     & (pts_coors[:,2].cpu().numpy()>=0) & (pts_coors[:,2].cpu().numpy()<=y_max) \
        #     & (pts_coors[:,3].cpu().numpy()>=0) & (pts_coors[:,3].cpu().numpy()<=x_max) 
        
        pts_coors[:,1] = pts_coors[:,1].clip(min=0,max=z_max)
        pts_coors[:,2] = pts_coors[:,2].clip(min=0,max=y_max)
        pts_coors[:,3] = pts_coors[:,3].clip(min=0,max=x_max)

        pts_pred = occupancy[:,:,pts_coors[:,1],pts_coors[:,2],pts_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu().numpy()

        # pts_pred[valid_mask==False]=15

        return pts_pred

    def voxelize(self, points,point_cloud_range,voxelization_size):
        """
        Input:
            points

        Output:
            coors [N,4]
            voxelized_data [M,3]
            voxel_coors [M,4]

        """
        voxel_size = torch.tensor(voxelization_size, device=points.device)
        pc_range = torch.tensor(point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)

        voxelized_data, voxel_coors = scatter_v2(points, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)

        batch_idx_pts = torch.zeros(coors.size(0),1).to(device=points.device)
        batch_idx_vox = torch.zeros(voxel_coors.size(0),1).to(device=points.device)

        coors_batch = torch.cat([batch_idx_pts,coors],dim=1)
        voxel_coors_batch = torch.cat([batch_idx_vox,voxel_coors],dim=1)

        return coors_batch.long(),voxelized_data,voxel_coors_batch.long()

    def decode_lidar_seg_hr(self,points,occupancy):

        out_h = 512
        out_w = 512
        out_z = 160

        self.voxel_lidar = [102.4/out_h,102.4/out_w,8/out_z]

        pts_coors,voxelized_data,voxel_coors = self.voxelize(points,self.pc_range,self.voxel_lidar)

        # clip out-ranged points
        z_max = int((self.pc_range[5]-self.pc_range[2])/self.voxel_lidar[2])-1
        y_max = int((self.pc_range[4]-self.pc_range[1])/self.voxel_lidar[1])-1
        x_max = int((self.pc_range[3]-self.pc_range[0])/self.voxel_lidar[0])-1
        pts_coors[:,1] = pts_coors[:,1].clip(min=0,max=z_max)
        pts_coors[:,2] = pts_coors[:,2].clip(min=0,max=y_max)
        pts_coors[:,3] = pts_coors[:,3].clip(min=0,max=x_max)


        new_h = torch.linspace(-1, 1, out_h).view(1,out_h,1).expand(out_z,out_h,out_w)
        new_w = torch.linspace(-1, 1, out_w).view(1,1,out_w).expand(out_z,out_h,out_w)
        new_z = torch.linspace(-1, 1, out_z).view(out_z,1,1).expand(out_z,out_h,out_w)

        grid = torch.cat((new_w.unsqueeze(3),new_h.unsqueeze(3), new_z.unsqueeze(3)), dim=-1)

        grid = grid.unsqueeze(0).to(occupancy.device)

        out_logit = F.grid_sample(occupancy, grid=grid)

        pts_pred = out_logit[:,:,pts_coors[:,1],pts_coors[:,2],pts_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu().numpy()
        return pts_pred

    def decode_occupancy(self,points,occupancy):
        out_h = 400
        out_w = 400
        out_z  = 64
        self.voxel_lidar = [102.4/out_h,102.4/out_w,8/out_z]

        pts_coors,voxelized_data,voxel_coors = self.voxelize(points,self.pc_range,self.voxel_lidar)


        # clip out-ranged points
        z_max = int((self.pc_range[5]-self.pc_range[2])/self.voxel_lidar[2])-1
        y_max = int((self.pc_range[4]-self.pc_range[1])/self.voxel_lidar[1])-1
        x_max = int((self.pc_range[3]-self.pc_range[0])/self.voxel_lidar[0])-1
        pts_coors[:,1] = pts_coors[:,1].clip(min=0,max=z_max)
        pts_coors[:,2] = pts_coors[:,2].clip(min=0,max=y_max)
        pts_coors[:,3] = pts_coors[:,3].clip(min=0,max=x_max)


        new_h = torch.linspace(-1, 1, out_h).view(1,out_h,1).expand(out_z,out_h,out_w)
        new_w = torch.linspace(-1, 1, out_w).view(1,1,out_w).expand(out_z,out_h,out_w)
        new_z = torch.linspace(-1, 1, out_z).view(out_z,1,1).expand(out_z,out_h,out_w)

        grid = torch.cat((new_w.unsqueeze(3),new_h.unsqueeze(3), new_z.unsqueeze(3)), dim=-1)

        grid = grid.unsqueeze(0).to(occupancy.device)

        out_logit = F.grid_sample(occupancy, grid=grid)

        # Occupancy Visualize
        out_class = out_logit.sigmoid()>0.2
        all_index = out_class.sum(dim=1).nonzero()

        out_voxel = out_logit[:,:,all_index[:,1],all_index[:,2],all_index[:,3]]
        out_voxel_scores = out_voxel.sigmoid()
        out_voxel_confidence,out_voxel_labels = out_voxel_scores.max(dim=1)
        output_occupancy = torch.cat((all_index.unsqueeze(0),out_voxel_labels.unsqueeze(-1)),dim=-1).cpu().numpy()[...,1:]

        return output_occupancy

    def decode_lidar_seg_dense(self, dense, occupancy):
        dense  = dense.long()
        pts_pred = occupancy[:,:,dense[0,:,0],dense[0,:,1],dense[0,:,2]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu().numpy()
        return pts_pred

    @torch.no_grad()
    def label_voxelization(self, pts_semantic_mask, pts_coors, voxel_coors):
        mask = pts_semantic_mask
        assert mask.size(0) == pts_coors.size(0)

        pts_coors_cls = torch.cat([pts_coors, mask], dim=1) #[N, 5]
        unq_coors_cls, unq_inv, unq_cnt = torch.unique(pts_coors_cls, return_inverse=True, return_counts=True, dim=0) #[N1, 5], [N], [N1]

        unq_coors, unq_inv_2, _ = torch.unique(unq_coors_cls[:, :4], return_inverse=True, return_counts=True, dim=0) #[N2, 4], [N1], [N2,]
        max_num, max_inds = torch_scatter.scatter_max(unq_cnt.float()[:,None], unq_inv_2, dim=0) #[N2, 1], [N2, 1]

        cls_of_max_num = unq_coors_cls[:, -1][max_inds.reshape(-1)] #[N2,]
        cls_of_max_num_N1 = cls_of_max_num[unq_inv_2] #[N1]
        cls_of_max_num_at_pts = cls_of_max_num_N1[unq_inv] #[N]

        assert cls_of_max_num_at_pts.size(0) == mask.size(0)

        cls_no_change = cls_of_max_num_at_pts == mask[:,0] # fix memory bug when scale up
        # cls_no_change = cls_of_max_num_at_pts == mask
        assert cls_no_change.any()

        max_pts_coors = pts_coors.max(0)[0]
        max_voxel_coors = voxel_coors.max(0)[0]
        assert (max_voxel_coors <= max_pts_coors).all()
        bsz, num_win_z, num_win_y, num_win_x = \
        int(max_pts_coors[0].item() + 1), int(max_pts_coors[1].item() + 1), int(max_pts_coors[2].item() + 1), int(max_pts_coors[3].item() + 1)

        canvas = -pts_coors.new_ones((bsz, num_win_z, num_win_y, num_win_x))

        canvas[pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2], pts_coors[:, 3]] = \
            torch.arange(pts_coors.size(0), dtype=pts_coors.dtype, device=pts_coors.device)

        fetch_inds_of_points = canvas[voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2], voxel_coors[:, 3]]

        assert (fetch_inds_of_points >= 0).all(), '-1 should not be in it.'

        voxel_label = cls_of_max_num_at_pts[fetch_inds_of_points]

        voxel_label = torch.clamp(voxel_label,min=0).long()

        return voxel_label

    @torch.no_grad()
    def get_point_pred(self,occupancy,pts_coors,voxel_coors,voxel_label,pts_semantic_mask):

        voxel_pred = occupancy[:,:,voxel_coors[:,1],voxel_coors[:,2],voxel_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu()

        voxel_gt = voxel_label.long().cpu()

        accurate = voxel_pred==voxel_gt

        acc = accurate.sum()/len(voxel_gt)

        pts_pred = occupancy[:,:,pts_coors[:,1],pts_coors[:,2],pts_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu()
        pts_gt  = pts_semantic_mask.long().squeeze(1).cpu()

        pts_accurate = pts_pred==pts_gt
        pts_acc = pts_accurate.sum()/len(pts_gt)

        return pts_acc
