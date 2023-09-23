import os
import pickle
from nuscenes import NuScenes
import argparse
import numpy as np
from tqdm import tqdm
import json

version = 'v1.0-test'
dataroot = '/data/nuscenes'
outdir = './test_panoocc'
verbose = True

os.makedirs(outdir, exist_ok=True)

outdir_lidarseg = os.path.join(outdir, 'lidarseg', 'test')
os.makedirs(outdir_lidarseg, exist_ok=True)

nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

test_pickle = './test_seg_result.pkl'

with open(test_pickle, 'rb') as f:
    test_result = pickle.load(f)

for test_file in tqdm(test_result):
    sample_token = test_file['token']
    sample = nusc.get('sample', sample_token)
    lidar_pred = test_file['lidar_pred']

    # processing 0 index for the submission
    new_pred = [15 if x == 0 else x for x in lidar_pred]

    lidar_sample_data_token = sample['data']['LIDAR_TOP']

    bin_file_path = os.path.join(outdir_lidarseg,lidar_sample_data_token + '_lidarseg.bin')

    np.array(new_pred).astype(np.uint8).tofile(bin_file_path)

    