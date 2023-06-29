from mmdet.datasets.builder import PIPELINES
import numpy as np
import os
from numpy import random
import mmcv
from mmcv.parallel import DataContainer as DC

@PIPELINES.register_module()
class LoadDenseLabel(object):
    def __init__(self,grid_size=[512, 512, 40], pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],occupancy_root=None):
        self.grid_size = np.array(grid_size)
        self.pc_range = pc_range
        self.occupancy_root = occupancy_root

    
    def __call__(self, results):

        scene_token = results['scene_token']
        lidar_token = results['lidar_token']

        occupancy_path = os.path.join(self.occupancy_root,'scene_'+scene_token,'occupancy',lidar_token+'.npy')
        
        # [z,y,x,label]
        occupancy_data = np.load(occupancy_path)

        results['dense_occupancy'] = occupancy_data

        return results



@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        # print(results.keys())
        # occ_gt_path = results['occ_gt_path']
        # occ_gt_path = os.path.join(self.data_root,occ_gt_path)

        # occ_labels = np.load(occ_gt_path)
        # semantics = occ_labels['semantics']
        # mask_lidar = occ_labels['mask_lidar']
        # mask_camera = occ_labels['mask_camera']
        if 'occ_gt_path' in results:
             occ_gt_path = results['occ_gt_path']
             occ_gt_path = os.path.join(self.data_root,occ_gt_path)

             occ_labels = np.load(occ_gt_path)
             semantics = occ_labels['semantics']
             mask_lidar = occ_labels['mask_lidar']
             mask_camera = occ_labels['mask_camera']
        else:
             semantics = np.zeros((200,200,16),dtype=np.uint8)
             mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
             mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)