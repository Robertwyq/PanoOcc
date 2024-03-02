import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time

@DETECTORS.register_module()
class PanoSegOcc(MVXTwoStageDetector):
    """PanoOcc.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 time_interval=1,
                 temporal_fuse_type="rnn",
                 HR_TEST = False,
                 DENSE_LABEL =False,
                 OCCUPANCY= False,
                 DENSE_EVAL=False,
                 ):

        super(PanoSegOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.time_interval = time_interval
        self.temporal_fuse_type = temporal_fuse_type
        self.HR_TEST = HR_TEST
        self.DENSE_LABEL = DENSE_LABEL
        self.OCCUPANCY = OCCUPANCY
        self.DENSE_EVAL = DENSE_EVAL

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': [],
            "ego2global_transform_lst": [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          pts_sem,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          dense_occupancy=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        if self.DENSE_LABEL:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, pts_sem, outs, dense_occupancy]
        else:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, pts_sem, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        prev_bev_lst = []
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]

                if self.temporal_fuse_type == "rnn":
                    if len(prev_bev_lst) > 0:
                        prev_bev = prev_bev_lst[-1]
                        # prev frame and current frame ego2global transformation
                        img_metas[0]["ego2global_transform_lst"] = img_metas_list[0][len_queue]["ego2global_transform_lst"][i-1:i+1]
                    else:
                        prev_bev = None
                elif self.temporal_fuse_type == "concat":
                    if len(prev_bev_lst) > 0:
                        prev_bev = torch.cat(prev_bev_lst, dim=1)
                        # all prev frame ego2global transformation
                        img_metas[0]["ego2global_transform_lst"] = img_metas_list[0][len_queue]["ego2global_transform_lst"][:i+1]
                    else:
                        prev_bev = None
                bev_feat, temporal_fused_bev_feat = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
                if self.temporal_fuse_type == "rnn":
                    prev_bev = temporal_fused_bev_feat
                elif self.temporal_fuse_type == "concat":
                    prev_bev = bev_feat
                prev_bev = prev_bev.permute(0, 2, 1)
                prev_bev = prev_bev.reshape(prev_bev.shape[0], 1, -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
                prev_bev_lst.append(prev_bev)
            self.train()
            # (bs, embed_dims, H, W)
            return prev_bev_lst

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      pts_semantic_mask= None,
                      dense_occupancy = None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        # Load Lidar semantic seg
        pts_sem = pts_semantic_mask[-1]

        # prev frame = 0, no temporal
        if prev_img.size(1)==0:
            prev_bev = None
        else:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        if self.temporal_fuse_type == "rnn":
            if prev_bev is not None and len(prev_bev) > 0:
                prev_bev = prev_bev[-1]
                # prev frame and current frame ego2global transformation
                img_metas[0][len_queue-1]["ego2global_transform_lst"] = img_metas[0][len_queue-1]["ego2global_transform_lst"][-2:]
            else:
                prev_bev = None
        elif self.temporal_fuse_type == "concat":
            if prev_bev is not None and len(prev_bev) > 0:
                prev_bev = torch.cat(prev_bev, dim=1)
            else:
                prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        if not self.DENSE_LABEL:
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, pts_sem, img_metas,
                                            gt_bboxes_ignore, prev_bev)
        else:
            assert dense_occupancy is not None
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, pts_sem, img_metas,
                                            gt_bboxes_ignore, prev_bev, dense_occupancy)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None,pts_semantic_mask=None, dense_occupancy=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0
        if points is not None:
            points = points[0]
        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[0][0].long().cpu().numpy()
            pts_semantic_mask = pts_semantic_mask[:,-1]
        if dense_occupancy is not None:
            dense_occupancy = dense_occupancy[0]

        self.prev_frame_info["ego2global_transformation_lst"].append(img_metas[0][0]["ego2global_transformation"])

        img_metas[0][0]["ego2global_transform_lst"] = self.prev_frame_info["ego2global_transformation_lst"][-1::-self.time_interval][::-1]
        if self.temporal_fuse_type == "concat":
            prev_bev = torch.cat(self.prev_frame_info["prev_bev"][-self.time_interval::-self.time_interval][::-1], dim=1) if len(self.prev_frame_info["prev_bev"]) > 0 else None
        elif self.temporal_fuse_type == "rnn":
            prev_bev = self.prev_frame_info["prev_bev"][-1] if len(self.prev_frame_info["prev_bev"]) > 0 else None
        if dense_occupancy is not None:
            prev_bev_feat, fused_prev_bev_feat, bbox_results, lidar_seg = self.simple_test(
                img_metas[0], img[0], points = points[0],
                prev_bev=prev_bev, dense_occupancy = dense_occupancy,
                **kwargs)
        else:
            prev_bev_feat, fused_prev_bev_feat, bbox_results, lidar_seg = self.simple_test(
                img_metas[0], img[0], points = points[0],
                prev_bev=prev_bev,
                **kwargs)

        if self.temporal_fuse_type == "concat":
            prev_bev = prev_bev_feat
        elif self.temporal_fuse_type == "rnn":
            prev_bev = fused_prev_bev_feat

        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        # (bs, H*W*Z, embed_dims) ->
        # (bs, num_queue, embed_dims, H, W, Z)
        prev_bev = prev_bev.permute(0, 2, 1).reshape(1, 1, -1, self.pts_bbox_head.bev_h,
                                                     self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
        self.prev_frame_info['prev_bev'].append(prev_bev)

        while len(self.prev_frame_info["ego2global_transformation_lst"]) >= self.pts_bbox_head.transformer.temporal_encoder.num_bev_queue * self.time_interval:
            self.prev_frame_info["ego2global_transformation_lst"].pop(0)
            self.prev_frame_info["prev_bev"].pop(0)
        if self.DENSE_EVAL:
            dense_label =  dense_occupancy[0,:,-1].cpu().numpy()
            lidar_results = dict(token= img_metas[0][0]['sample_idx'],lidar_pred = lidar_seg, lidar_label = dense_label)
        else:
            lidar_results = dict(token= img_metas[0][0]['sample_idx'],lidar_pred = lidar_seg, lidar_label = pts_semantic_mask)

        return bbox_results, lidar_results

    def simple_test_pts(self, x, img_metas, points=None, prev_bev=None, dense_occupancy=None,rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        if self.OCCUPANCY:
            occupancy = self.pts_bbox_head.decode_occupancy(points,outs['occupancy'])
            bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
            ]
            return outs["bev_feat"], outs['bev_embed'], bbox_results, occupancy
        else:
            if self.DENSE_EVAL:
                lidar_seg = self.pts_bbox_head.decode_lidar_seg_dense(dense_occupancy,outs['occupancy'])
            else:
                if self.HR_TEST:
                    lidar_seg = self.pts_bbox_head.decode_lidar_seg_hr(points,outs['occupancy'])
                else:
                    lidar_seg = self.pts_bbox_head.decode_lidar_seg(points,outs['occupancy'])

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_feat"], outs['bev_embed'], bbox_results, lidar_seg

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, dense_occupancy=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        prev_bev_feat, fused_prev_bev_feat, bbox_pts, lidar_seg = self.simple_test_pts(img_feats, img_metas, points, prev_bev,dense_occupancy = dense_occupancy, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return prev_bev_feat, fused_prev_bev_feat, bbox_list, lidar_seg



