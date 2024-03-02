# NuScenes
Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). 
## 1. NuScenes LiDAR Benchmark
- Train by supervision from LiDAR points, and evaluation on the LiDAR benchmark.
```shell
# For segmentation & panoptic segmentation, You need lidarseg & panoptic
data_path: ./data/nuscenes
# cp yaml file for label mapping
cp projects/configs/label_mapping/nuscenes.yaml ./data/nuscenes/
```
**dataset structure**
```
nuscenes
├── can_bus/
├── maps/
├── lidarseg/
├── panoptic/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── v1.0-test/
├── nuscenes_infos_temporal_train.pkl
├── nuscenes_infos_temporal_val.pkl
├── nuscenes_infos_temporal_test.pkl
├── nuscenes.yaml
```

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

## 2. NuScenes Occupancy Benchmark (CVPR2023 workshop)
Download the annotations [HERE](https://opendatalab.com/CVPR2023-3D-Occupancy/cli)
- download gts & annotations is enough, no need for the img, others are same in the nuScenes.

**dataset structure**
```
occ3d-nus
├── gts/
├── maps/
├── lidarseg/
├── panoptic/
├── samples/
├── v1.0-trainval/
├── occ_infos_temporal_train.pkl
├── occ_infos_temporal_val.pkl

├── occ3d-test/
│ ├── maps/
│ ├── samples/
│ ├── v1.0-test/
│ └── annotations.json
```

Generate the info files for training and validation:
```
python tools/create_data_occ.py occ --root-path ./data/occ3d-nus --out-dir ./data/occ3d-nus --extra-tag occ --version v1.0-trainval --canbus ./data/nuscenes --occ-path ./data/occ3d-nus
``` 
Generate the info files for trainval split:
```
# modify the path
python tools/merge_data_occ.py
```

Generate the info files for test split:
```
python tools/create_data_occ.py occ --root-path ./data/occ3d-test --out-dir ./data/occ3d-test --extra-tag occ --version v1.0-test --canbus ./data/nuscenes --occ-path ./data/occ3d-test
```

## 3. Semantic KITTI
To prepare for SemanticKITTI dataset, please download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download).

## 4. NuScenes OpenOccupancy
refer to https://github.com/JeffWang987/OpenOccupancy

we only use the dense label:
```
mv nuScenes-Occupancy-v0.1.7z ./data
cd ./data
7za x nuScenes-Occupancy-v0.1.7z
mv nuScenes-Occupancy-v0.1 nuScenes-Occupancy
```