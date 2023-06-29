# NuScenes

## 1. NuScenes Occupancy Benchmark (CVPR workshop)
**a. Train PanoOcc with 8 GPUs.**
```shell
./tools/dist_train.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/PanoOcc_small.py 8
```

**b. Test PanoOcc with 8 GPUs.**
```shell
./tools/dist_test_dense.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/PanoOcc_small.py work_dirs/PanoOcc_small/epoch_24.pth 8
```
You can evaluate the F-score at the same time by adding `--eval_fscore`.

**d. Test with 8 GPUs for test split.**
```shell
./tools/dist_test_dense.sh ./projects/configs/PanoOcc/Occupancy/Occ3d-nuScenes/PanoOcc_small.py work_dirs/PanoOcc_small/epoch_24.pth 8 --format-only --eval-options 'submission_prefix=./occ_submission'
 ```