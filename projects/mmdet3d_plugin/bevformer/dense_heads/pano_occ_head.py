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
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet3d.core import bbox3d2result, LiDARInstance3DBoxes

@HEADS.register_module()
class PanoOccHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 sync_cls_avg_factor=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 code_weights=None,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 bev_z=5,
                 loss_occ=None,
                 use_mask=False,
                 with_det = False,
                 num_query = 900,
                 loss_cls = None,
                 loss_bbox = None,
                 loss_iou = None,
                 assigner = None,
                 loss_occupancy_aux = None,
                 loss_det_occ = None,
                 positional_encoding=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_occupancy_aux = loss_occupancy_aux
        self.loss_det_occ = loss_det_occ
        
        self.with_det = with_det
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage


        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(PanoOccHead, self).__init__()

        self.loss_occ = build_loss(loss_occ)
        if loss_occupancy_aux is not None:
            self.aux_loss = build_loss(loss_occupancy_aux)
        if loss_det_occ is not None:
            self.det_occ_loss = build_loss(loss_det_occ)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
            
        if self.with_det:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            self.code_weights = nn.Parameter(torch.tensor(
                    self.code_weights, requires_grad=False), requires_grad=False)
            self.bbox_coder = build_bbox_coder(bbox_coder)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, 10))
            fc_cls = nn.Sequential(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, 10))
            reg_branch = nn.Sequential(*reg_branch)
            num_pred = (self.transformer.decoder.num_layers + 1) if \
                    self.as_two_stage else self.transformer.decoder.num_layers
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

            self.loss_cls = build_loss(loss_cls)
            self.loss_bbox = build_loss(loss_bbox)
            self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.with_det:
            if self.loss_cls.use_sigmoid:
                bias_init = bias_init_with_prob(0.01)
                for m in self.cls_branches:
                    nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, test=False):
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
        if self.with_det:
            object_query_embeds = self.query_embedding.weight.to(dtype)
        else:
            object_query_embeds = None
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w, self.bev_z),device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
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
                reg_branches=None, 
                cls_branches=None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
        if self.with_det:
            bev_embed, occ_outs, voxel_det, hs, init_reference, inter_references = outputs
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
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'occ': occ_outs,
                'det_occ':voxel_det,
            }
        else:
            bev_embed, occ_outs = outputs

            outs = {
                'bev_embed': bev_embed,
                'occ':occ_outs,
            }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()

        occ=preds_dicts['occ']

        if self.with_det:
            all_cls_scores = preds_dicts['all_cls_scores']
            all_bbox_preds = preds_dicts['all_bbox_preds']
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
                self.loss_single_det, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

            loss_dict['loss_cls'] = losses_cls[-1]
            loss_dict['loss_bbox'] = losses_bbox[-1]

            # add det occ
            det_occ = preds_dicts['det_occ']
            voxel_det_mask = torch.ones(1,det_occ.shape[1],det_occ.shape[2],det_occ.shape[3]).to(occ.device).long()
            voxel_det = (voxel_semantics<17) & mask_camera.to(bool)
            Idx = torch.where(voxel_det)
            voxel_det_mask[Idx[0],Idx[1],Idx[2],Idx[3]]= 0
            if self.loss_det_occ is not None:
                num_total_pos = voxel_det.sum()
                avg_factor = num_total_pos*1.0
                if self.sync_cls_avg_factor:
                    avg_factor = reduce_mean(
                        det_occ.new_tensor([avg_factor]))
                avg_factor = max(avg_factor, 1)
                losses_det_occ = self.det_occ_loss(det_occ.reshape(-1, 1),voxel_det_mask.reshape(-1),avg_factor=avg_factor)
                loss_dict['loss_det_occ']=losses_det_occ
            
            # voxel_det_mask = torch.zeros(1,det_occ.shape[1],det_occ.shape[2],det_occ.shape[3]).to(occ.device).long()
            # voxel_det = ((voxel_semantics>=1) & (voxel_semantics<=10))
            # Idx = torch.where(voxel_det)
            # voxel_det_mask[Idx[0],Idx[1],Idx[2],Idx[3]]= 1
            # empty_mask = voxel_semantics==17
            # empty_Idx = torch.where(empty_mask)
            # voxel_det_mask[empty_Idx[0],empty_Idx[1],empty_Idx[2],empty_Idx[3]]= 2 
            # if self.loss_det_occ is not None:
            #     num_total_pos = (voxel_semantics<17).sum()
            #     # num_total_pos= ((voxel_semantics<17) & mask_camera.to(bool)).sum()
            #     avg_factor = num_total_pos*1.0
            #     if self.sync_cls_avg_factor:
            #         avg_factor = reduce_mean(
            #             det_occ.new_tensor([avg_factor]))
            #     avg_factor = max(avg_factor, 1)
            #     losses_det_occ = self.det_occ_loss(det_occ.reshape(-1, 2),voxel_det_mask.reshape(-1),avg_factor=avg_factor)
            #     loss_dict['loss_det_occ']=losses_det_occ
            
        
        assert voxel_semantics.min()>=0 and voxel_semantics.max()<=17

        losses = self.loss_single(voxel_semantics,mask_camera,occ)
        if self.loss_occupancy_aux is not None:
            occ_aux = occ.permute(0,4,1,2,3)
            losses_aux = self.aux_loss(occ_aux,voxel_semantics)
            loss_dict['loss_occ_aux']=losses_aux
        loss_dict['loss_occ']=losses
        return loss_dict

    def loss_single(self,voxel_semantics,mask_camera,preds):
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)

            # focal loss
            # num_total_pos = (voxel_semantics<17).sum()
            # avg_factor = num_total_pos*1.0
            # if self.sync_cls_avg_factor:
            #     avg_factor = reduce_mean(
            #         preds.new_tensor([avg_factor]))
            # avg_factor = max(avg_factor, 1)
            # loss_occ = self.loss_occ(preds, voxel_semantics,avg_factor=avg_factor)

            # ce loss
            loss_occ = self.loss_occ(preds, voxel_semantics)
        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)

        # # Post processing 1
        det_occ = preds_dicts['det_occ']
        occupancy_mask = det_occ.sigmoid() < 0.05
        occupancy_mask = occupancy_mask.squeeze(-1)
        occ_score[occupancy_mask==1] = 17

        # Post processing 2
        
        # if self.with_det:
        #     bbox_list = self.get_bboxes(preds_dicts, img_metas, rescale=rescale)
        #     bbox_results = [
        #         bbox3d2result(bboxes, scores, labels)
        #             for bboxes, scores, labels in bbox_list
        #     ]
        #     bbox_3d = bbox_results[0]['boxes_3d']
        #     bbox_3d = bbox_3d.tensor[:,:7]
        #     bbox_3d = LiDARInstance3DBoxes(bbox_3d,box_dim=bbox_3d.shape[-1]).convert_to(0)
        #     grid  = self.get_voxel_grid()
        #     pc_semantic = occ_score.cpu().numpy()
        #     pred_foreground = np.where((pc_semantic[:]<=10) &(pc_semantic[:]>0),1,0)
        #     new_pred_semantics = pc_semantic.copy()

        #     if pred_foreground.sum()>0:
        #         pc = grid[pred_foreground==1][:,:3].cuda()
        #         box_valid = bbox_3d.enlarged_box(2.0).points_in_boxes(pc).cpu().numpy()>-1
        #         if box_valid.sum()>0:
        #             remove_x = pred_foreground.nonzero()[1][box_valid==0]
        #             remove_y = pred_foreground.nonzero()[2][box_valid==0]
        #             remove_z = pred_foreground.nonzero()[3][box_valid==0]
        #             new_pred_semantics[:,remove_x,remove_y,remove_z]=17
        
        #     return torch.tensor(new_pred_semantics).to(occ_score.device)

        return occ_score
        

    def loss_single_det(self,
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
        cls_scores = cls_scores.reshape(-1, 10)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * 0.0
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
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
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
                                    10,
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

    def get_voxel_grid(self, bev_h=200,bev_w=200,bev_z=16):
        pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        ref_x, ref_y, ref_z = torch.meshgrid(
                        torch.linspace(0.5, bev_h - 0.5, bev_h),
                        torch.linspace(0.5, bev_w - 0.5, bev_w),
                        torch.linspace(0.5, bev_z - 0.5, bev_z),
                    )
        ref_y = ref_y / bev_h
        ref_x = ref_x / bev_w
        ref_z = ref_z / bev_z
        grid = torch.stack(
                        (ref_x,
                        ref_y,
                        ref_z,
                        ref_x.new_ones(ref_x.shape)), dim=-1)
        min_x, min_y, min_z, max_x, max_y, max_z = pc_range
        grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
        grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
        grid[..., 2] = grid[..., 2] * (max_z - min_z) + min_z
        return grid