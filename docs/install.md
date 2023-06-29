# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n occ python=3.8 -y
conda activate occ
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install mmcv series.**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**d. Install others.**
```shell
pip install timm
pip install einops
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
```

**e. Install PanoOcc.**
```shell
git clone https://github.com/Robertwyq/PanoOcc

cd mmdetection3d 
pip install -v -e . (python setup.py install)
# for InternImage
cd ops 
pip install -v -e .
```

**Other Problems Maybe Meet**
```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt-get install make gcc g++
pip install ninja
sudo apt-get install libgl1-mesa-dev
pip install setuptools==56.1.0
```