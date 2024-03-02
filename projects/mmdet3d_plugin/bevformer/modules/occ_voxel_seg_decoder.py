from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VoxelNaiveDecoder(BaseModule):

    def __init__(self,
                 bev_h=50,
                 bev_w=50,
                 bev_z=8,
                 conv_up_layer = 2,
                 embed_dim = 256,
                 out_dim = 64,):
        super(VoxelNaiveDecoder, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = out_dim
        self.conv_up_layer = conv_up_layer
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim,embed_dim,(1,5,5),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, embed_dim, (1, 4, 4), stride=(1, 2, 2), padding=(0,1,1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, self.out_dim, (2, 4, 4), stride=(2, 2, 2),padding=(0,1,1)),
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.semantic_det1 = nn.Sequential(nn.Conv3d(embed_dim, 2, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
                
    def forward(self, inputs):
        
        voxel_input = inputs.view(1,self.bev_h, self.bev_w, self.bev_z, -1).permute(0,4,3,1,2)

        voxel_det1 = self.semantic_det1(voxel_input)

        voxel_feat = self.upsample(voxel_input)

        voxel_det = [voxel_det1]

        return voxel_feat, voxel_det