# NuScenes

## 1. NuScenes Occupancy Benchmark (CVPR2023 workshop)
**a. Train PanoOcc with 8 GPUs.**
```shell
./tools/dist_train.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/PanoOcc_small.py 8
```
**b. Test PanoOcc with 8 GPUs.**
```shell
./tools/dist_test_dense.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/PanoOcc_small.py work_dirs/PanoOcc_small/epoch_24.pth 8
```
You can evaluate the F-score at the same time by adding `--eval_fscore`.

**c. Test with 8 GPUs for test split.**
```shell
./tools/dist_test_dense.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/PanoOcc_small.py work_dirs/PanoOcc_small/epoch_24.pth 8 --format-only --eval-options 'submission_prefix=./occ_submission'
 ```

 ## 2. NuScenes Lidar Segmentation Benchmark
 **a. Train PanoOcc with 8 GPUs.**
```shell
./tools/dist_train.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_small_4f.py 8
```
**b. Test PanoOcc with 8 GPUs.**
```shell
# for segmentation eval
./tools/dist_test_seg.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_small_4f.py work_dirs/PanoOcc_small_4f/epoch_24.pth 8
# for detection eval
./tools/dist_test.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_small_4f.py work_dirs/PanoOcc_small_4f/epoch_24.pth 8
```

**c. Test with 8 GPUs for test split.**
```shell
./tools/dist_test_seg_submit.sh ./projects/configs/PanoOcc/Panoptic/PanoOcc_base_4f_cat_test.py work_dirs/PanoOcc_base_4f_cat_test/epoch_24.pth 8

# Also need to convert to submission format, remind to add submission.json in the folder
python ./tools/test_submit.py
```