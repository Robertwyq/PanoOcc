from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F
import numpy as np

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VoxelDecoder(BaseModule):

    def __init__(self,
                 bev_h=50,
                 bev_w=50,
                 bev_z=8,
                 conv_up_layer = 2,
                 embed_dim = 256,
                 out_dim = 64,):
        super(VoxelDecoder, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = out_dim
        self.conv_up_layer = conv_up_layer
        upsample_scale = int(np.math.log2(200 // bev_h))
        upsample = []
        if upsample_scale == 0:
            upsample.append(nn.ConvTranspose3d(embed_dim, self.out_dim, (1, 3, 3), stride=(1, 1, 1),padding=(0,1,1)))
            upsample.append(nn.BatchNorm3d(self.out_dim))
            upsample.append(nn.ReLU(inplace=True))
        else:
            for _ in range(upsample_scale-1):
                upsample.append(nn.ConvTranspose3d(embed_dim, embed_dim, (1, 4, 4), stride=(1, 2, 2), padding=(0,1,1)))
                upsample.append(nn.BatchNorm3d(embed_dim))
                upsample.append(nn.ReLU(inplace=True))

            upsample.append(nn.ConvTranspose3d(embed_dim, self.out_dim, (1, 4, 4), stride=(1, 2, 2),padding=(0,1,1)))
            upsample.append(nn.BatchNorm3d(self.out_dim))
            upsample.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*upsample)

        self.semantic_det = nn.Sequential(nn.Conv3d(self.out_dim, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, inputs):
        
        voxel_input = inputs.view(1,self.bev_h,self.bev_w,self.bev_z, -1).permute(0,4,3,1,2)

        voxel_feat = self.upsample(voxel_input)

        voxel_det = self.semantic_det(voxel_feat)
        
        return voxel_feat,voxel_det