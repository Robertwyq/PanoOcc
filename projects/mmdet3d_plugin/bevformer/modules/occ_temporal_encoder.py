
from mmcv.cnn.bricks.registry import ATTENTION
import torch
from torch import nn
import torch.utils.checkpoint as cp


from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn import kaiming_init, constant_init
from mmcv.runner.base_module import BaseModule, Sequential

from .residual_block_3d import Bottleneck
from .x3d_block import ResBlock, X3DTransform
from .occ_temporal_attention import OccTemporalAttention



@ATTENTION.register_module()
class OccTemporalEncoder(BaseModule):
    def __init__(self, bev_h: int, bev_w: int, bev_z: int, num_bev_queue: int, embed_dims: int, num_block: int,
                 block_type: str, conv_cfg, norm_cfg, init_cfg=None) ->None:
        super().__init__(init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_bev_queue = num_bev_queue
        self.embed_dims = embed_dims
        self.block_type = block_type
        in_channels = num_bev_queue * embed_dims
        out_channels = embed_dims

        if block_type == "x3d":
            temporal_block = [ResBlock(in_channels, in_channels, 3, 1,  trans_func=X3DTransform, dim_inner=in_channels//4) for _ in range(num_block-1)]
            temporal_block.append(ResBlock(in_channels, out_channels, 3, 1,  trans_func=X3DTransform, dim_inner=in_channels//4))
            self.temporal_block = nn.Sequential(*temporal_block)
        elif block_type == "c3d":
            temporal_block = [Bottleneck(in_channels, in_channels//4, conv_cfg=conv_cfg, norm_cfg=norm_cfg) for _ in range(num_block-1)]
            temporal_block.append(Bottleneck(in_channels, out_channels, downsample=nn.Sequential(
                build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ), conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            self.temporal_block = nn.Sequential(*temporal_block)

        elif block_type == "self_attn":
            self.temporal_block = OccTemporalAttention(embed_dims=embed_dims, num_levels=1, num_bev_queue=num_bev_queue)

    def init_weights(self):
        if self.block_type == "self_attn":
            self.temporal_block.init_weights()
        else:
            for module in self.modules():
                if isinstance(module, nn.Conv3d):
                    kaiming_init(module)
                elif isinstance(module, nn.BatchNorm3d):
                    constant_init(module, 1)

    def forward(self, bev_feat: torch.Tensor, prev_bev: torch.Tensor = None, ref_2d: torch.Tensor = None, bev_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the BEV temporal encoder.

        Args:
            bev_feat (torch.Tensor): Input tensor of shape (bs, embed_dims, H, W), where
                bs is the batch size, embed_dims is the number of feature dimensions,
                H is the height and W is the width of the bev feat.
            prev_bev (torch.Tensor, optional): Tensor representing the previous BEV features.
                If not provided, it is set to None (default). The shape of the tensor
                is (bs, num_queue-1, dim, H, W), where num_queue is the number of previous frames
                to be stored in a queue and dim is the number of feature dimensions.

        Returns:
            temporal_fused_bev_feat (torch.Tensor): Output tensor with shape (bs, H*W, embed_dims),
            representing the temporally fused BEV features after passing through
            the `TemporalEncoder` module.
        """
        # Change bev_feat shape to [bs, embed_dims, H, W, Z]
        bev_feat = bev_feat.permute(0, 2, 1)
        bev_feat = bev_feat.reshape(bev_feat.shape[0], -1, self.bev_h, self.bev_w, self.bev_z)
        assert bev_feat.shape[1] == self.embed_dims, "bev features dims do not match!"

        if prev_bev is None:
            # first frame has no prev_bev
            prev_bev = torch.cat(
                [bev_feat for _ in range(self.num_bev_queue - 1)], dim=1
            )  # [bs, (num_queue-1)*embed_dims, H, W]
        else:
            padding_bev = [
                prev_bev[:, 0:1] for _ in range(prev_bev.shape[1], self.num_bev_queue - 1)
            ]
            prev_bev = torch.cat([*padding_bev, prev_bev], dim=1)  # [bs, num_queue-1, dim, H, W]
            prev_bev = prev_bev.reshape(
                prev_bev.shape[0], -1, self.bev_h, self.bev_w, self.bev_z
            )  # [bs, (num_queue-1)*embed_dims, H, W, Z]

        if self.block_type == "self_attn":
            # change query/value shape to [bs, H*W*Z, embed_dims]
            bev_feat = bev_feat.permute(0, 2, 3, 4, 1).reshape(bev_feat.shape[0], -1, bev_feat.shape[1])
            prev_bev = prev_bev.permute(0, 2, 3, 4, 1).reshape(prev_bev.shape[0], -1, prev_bev.shape[1])
            bs = ref_2d.shape[0]
            prev_bev = torch.stack([prev_bev, bev_feat], dim=1).reshape(bs*2, -1, self.embed_dims)
            temporal_fused_bev_feat = self.temporal_block(query=bev_feat, value=prev_bev,
                                                          query_pos=bev_pos,
                                                          reference_points=ref_2d,
                                                          spatial_shapes=torch.tensor(
                                                              [[self.bev_h, self.bev_w, self.bev_z]], device=bev_feat.device),
                                                          level_start_index=torch.tensor([0], device=bev_feat.device),
                                                          )
        else:
            # bev_queue with shape (bs, num_queue*embed_dims, H, W, Z)
            bev_queue = torch.cat([prev_bev, bev_feat], dim=1)

            bev_queue = bev_queue.permute(0, 1, 4, 2, 3)  # [bs, dim, Z, H, W]
            temporal_fused_bev_feat = self.temporal_block(bev_queue)  # (bs, embed_dims, Z, H, W)
            temporal_fused_bev_feat = temporal_fused_bev_feat.permute(0, 1, 3, 4, 2)  # (bs, embed_dims, H, W, Z)
            temporal_fused_bev_feat = temporal_fused_bev_feat.reshape(
                temporal_fused_bev_feat.shape[0], self.embed_dims, -1)
            temporal_fused_bev_feat = temporal_fused_bev_feat.permute(0, 2, 1)  # (bs, H*W*Z, embed_dims)

        return temporal_fused_bev_feat
