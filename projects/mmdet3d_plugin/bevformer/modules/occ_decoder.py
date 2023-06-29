from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccupancyDecoder(BaseModule):

    def __init__(self,
                 num_classes,
                 bev_h=50,
                 bev_w=50,
                 bev_z=8,
                 conv_up_layer = 2,
                 inter_up_rate = [1,2,2],
                 embed_dim = 256,
                 upsampling_method='trilinear',
                 align_corners=False):
        super(OccupancyDecoder, self).__init__()
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = embed_dim // 2
        self.align_corners = align_corners
        self.inter_up_rate = inter_up_rate
        self.conv_up_layer = conv_up_layer
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim,embed_dim,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, embed_dim, (2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, self.out_dim, (2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.semantic_det = nn.Sequential(nn.Conv3d(embed_dim, 2, kernel_size=1))
        self.semantic_cls = nn.Sequential(nn.Conv3d(self.out_dim, self.num_classes,kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    

    def forward(self, inputs):
        
        # z x y
        voxel_input = inputs.view(1,self.bev_w,self.bev_h,self.bev_z, -1).permute(0,4,3,1,2)

        voxel_det = self.semantic_det(voxel_input)

        voxel_up1 = self.upsample(voxel_input)

        voxel_cls = self.semantic_cls(voxel_up1)

        voxel_pred = F.interpolate(voxel_cls,scale_factor=(self.inter_up_rate[0],self.inter_up_rate[1],self.inter_up_rate[2]),mode=self.upsampling_method,align_corners=self.align_corners)

        return voxel_pred, voxel_det