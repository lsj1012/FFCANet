import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class InjectionMultiSum(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.norm_cfg = norm_cfg
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
        self.ca = ChannelAttention(oup)
        self.cat_conv = ConvModule(inp+oup, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * sig_act + global_feat
        out = self.ca(out)
        out = self.cat_conv(torch.cat([out,x_l],dim=1))
        return out