import json
import numpy as np
import pickle
from nuscenes import NuScenes
import os

def load_pkl(path):
    f = open(path,'rb')
    info = pickle.load(f)
    return info

def main():
    nuscenes_root = '/data/yuqi_wang/nuscenes' 
    pred_lidarseg_path = '/root/workspace/Occupancy/seg_result.pkl'
    output_dir = '/root/workspace/Occupancy/work_dirs/lidar_seg_r101'
    os.makedirs(output_dir,exist_ok=True)

    val_path = os.path.join(nuscenes_root,'nuscenes_infos_temporal_val.pkl')
    nusc_seg = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=True)

    val_info = load_pkl(val_path)
    pred_info = load_pkl(pred_lidarseg_path)

    pred_seg = {}
    for p in pred_info:
        pred_seg[p['token']]=p['lidar_pred']
    
    for vi in range(len(val_info['infos'])):
        vif = val_info['infos'][vi]
        lidar_sd_token = nusc_seg.get('sample', vif['token'])['data']['LIDAR_TOP']
        save_name = lidar_sd_token+'_lidarseg.bin'
        save_path = os.path.join(output_dir,save_name)
        pred = pred_seg[vif['token']].astype('uint8')
        assert pred.shape
        pred.tofile(save_path)
        if vi%500==0:
            print(vi)


if __name__ == '__main__':
    main()