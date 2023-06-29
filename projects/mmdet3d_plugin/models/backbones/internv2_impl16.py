import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from collections import OrderedDict
import warnings
import math
import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.utils import get_root_logger
from ops.modules import MSDeformAttnGrid_final_softmax as MSDeformAttn


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# @FEEDFORWARD_NETWORK.register_module()
# class Mlp(BaseModule):


#     def __init__(self,
#                  in_features=256,
#                  hidden_features=1024,
#                  num_fcs=2,
#                  act_layer=nn.GELU,
#                  drop=0.,
#                  act_cfg=dict(type='ReLU', inplace=True),
#                  ffn_drop=0.,
#                  dropout_layer=None,
#                  add_identity=True,
#                  init_cfg=None,
#                  split=4,
#                  use_checkpoint=False,
#                  **kwargs):
#         super(Mlp, self).__init__(init_cfg)
#         assert num_fcs >= 2, 'num_fcs should be no less ' \
#             f'than 2. got {num_fcs}.'
#         embed_dims = in_features
#         feedforward_channels = hidden_features
#         ffn_drop = drop

#         self.embed_dims = embed_dims
#         self.feedforward_channels = feedforward_channels
#         self.num_fcs = num_fcs
#         self.act_cfg = act_cfg
#         # self.activate = build_activation_layer(act_cfg)
#         self.activate = act_layer()
#         self.drop = nn.Dropout(ffn_drop)
#         in_channels = embed_dims
#         self.use_checkpoint = use_checkpoint
#         self.split = split
#         for i in range(split):
#             fc1 = nn.Linear(in_channels, feedforward_channels //
#                             self.split, bias=True)
#             setattr(self, f"fc1_{i}", fc1)

#         for i in range(split):
#             fc2 = nn.Linear(feedforward_channels // self.split,
#                             embed_dims, bias=False)
#             setattr(self, f"fc2_{i}", fc2)
#         self.fc2_bias = nn.Parameter(torch.zeros(
#             (embed_dims)), requires_grad=True)
#         self.dropout_layer = build_dropout(
#             dropout_layer) if dropout_layer else torch.nn.Identity()
#         self.add_identity = add_identity

#     def forward(self, x, identity=None):

#         def _inner_forward(x, i):
#             fc1 = getattr(self, f"fc1_{i}")
#             x = fc1(x)
#             x = self.activate(x)
#             x = self.drop(x)
#             fc2 = getattr(self, f"fc2_{i}")
#             x = fc2(x)
#             x = self.drop(x)
#             return x

#         out = 0
#         for i in range(self.split):
#             if self.use_checkpoint and x.requires_grad:
#                 out = out + checkpoint.checkpoint(_inner_forward, x, i)
#             else:
#                 out = out + _inner_forward(x, i)

#         out += self.fc2_bias

#         if not self.add_identity:
#             return self.dropout_layer(out)
#         if identity is None:
#             identity = x
#         return identity + self.dropout_layer(out)



class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7, deform_points=25, deform_ratio=1.0,
                 dilation_rates=[1, 2, 3], deform_padding=True, use_hw_scaler=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # self.attn = NeighborhoodAttention(
        #     dim, kernel_size=kernel_size, num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn = MSDeformAttn(d_model=dim, n_levels=1, n_heads=num_heads,
                                 n_points=deform_points, ratio=deform_ratio, dilation_rates=dilation_rates,
                                 padding=deform_padding, dw_ks=kernel_size,
                                 use_hw_scaler=use_hw_scaler)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, deform_inputs):
        def deform_forward(x):
            n, h, w, c = x.shape
            x = self.attn(
                query=x.view(n, h * w, c),
                reference_points=deform_inputs[0],
                input_flatten=None,
                input_spatial_shapes=deform_inputs[1],
                input_level_start_index=deform_inputs[2],
                input_padding_mask=None).view(n, h, w, c)
            return x

        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = deform_forward(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x

        x = self.norm1(x)
        x = deform_forward(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 deform_points=4, deform_padding=True, dilation_rates=[1, 2, 3],
                 drop_path=0., norm_layer=nn.LayerNorm, use_hw_scaler=None,
                 layer_scale=None, with_cp=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.with_cp = with_cp
        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     deform_points=deform_points,
                     deform_padding=deform_padding,
                     dilation_rates=dilation_rates,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, layer_scale=layer_scale,
                     use_hw_scaler=use_hw_scaler)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x, deform_inputs):
        for blk in self.blocks:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(blk, x, deform_inputs)
            else:
                x = blk(x, deform_inputs)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


@BACKBONES.register_module()
class InternV2Impl16(BaseModule):
    def __init__(self, embed_dim=64, mlp_ratio=3., depths=[3, 4, 18, 5], num_heads=[3, 6, 12, 24],
                 drop_path_rate=0.2, in_chans=3, kernel_size=5, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, layer_scale=None,
                 deform_points=25, deform_padding=True, dilation_rates=[1], init_cfg=None,
                 pretrained=None, norm_cfg=dict(type='LN'), out_indices=(0, 1, 2, 3),
                 use_hw_scaler=None, with_cp=False, cp_level=0, **kwargs):

        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.out_indices = out_indices
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        print(self.num_features)
        self.mlp_ratio = mlp_ratio
        self.deform_padding = deform_padding
        self.deform_points = deform_points
        print("deform padding:", deform_padding)
        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),
                             layer_scale=layer_scale,
                             dilation_rates=dilation_rates,
                             deform_points=deform_points,
                             deform_padding=deform_padding,
                             use_hw_scaler=use_hw_scaler,
                             with_cp=with_cp if i < cp_level else False)
            self.levels.append(level)
            if i < self.num_levels - 1:
                norm = norm_layer(int(embed_dim * 2 ** (i + 1)))
                setattr(self, f"norm{i + 1}", norm)

        self.num_layers = len(depths)
        for i in self.out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'final_norm{i}'
            self.add_module(layer_name, layer)

        # self.apply(self._init_weights)
        # self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            self.apply(self._init_deform_weights)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            meg = self.load_state_dict(state_dict, False)
            logger.info(meg)

    def _get_reference_points(self, spatial_shapes, device, padding=0):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(padding + 0.5, H_ - padding - 0.5,
                               int(H_ - 2 * padding),
                               dtype=torch.float32, device=device),
                torch.linspace(padding + 0.5, W_ - padding - 0.5,
                               int(W_ - 2 * padding),
                               dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def _deform_inputs(self, x):
        bs, c, h, w = x.shape
        deform_inputs = list()
        if self.deform_padding:
            padding = int(math.sqrt(self.deform_points) // 2)
        else:
            padding = int(0)

        for i in range(self.num_layers):
            spatial_shapes = torch.as_tensor(
                [(h // pow(2, i + 2) + 2 * padding,
                  w // pow(2, i + 2) + 2 * padding)],
                dtype=torch.long, device=x.device)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)),
                 spatial_shapes.prod(1).cumsum(0)[:-1]))
            reference_points = self._get_reference_points(
                [(h // pow(2, i + 2) + 2 * padding,
                  w // pow(2, i + 2) + 2 * padding)],
                device=x.device, padding=padding)
            deform_inputs.append(
                [reference_points, spatial_shapes, level_start_index,
                 (h // pow(2, i + 2), w // pow(2, i + 2))])
        return deform_inputs

    def forward(self, x):
        deform_inputs = self._deform_inputs(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        outs = []
        for i, level in enumerate(self.levels):
            x, xo = level(x, deform_inputs[i])
            if i != self.num_levels - 1:
                norm = getattr(self, f'norm{i + 1}')
                x = norm(x)
            if i in self.out_indices:
                final_norm = getattr(self, f'final_norm{i}')
                xo = final_norm(xo)
                outs.append(xo.permute(0, 3, 1, 2).contiguous())
        return outs


