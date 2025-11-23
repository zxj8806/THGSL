import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d as TVDeformConv2d
except Exception as e:
    TVDeformConv2d = None

try:
    from torchvision.ops import DeformConv2d
except Exception as e:
    raise ImportError(
        "TorchVision DeformConv2d not available. "
        "Please `pip install --upgrade torchvision` matching your torch version."
    ) from e


class DCN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 groups=1,
                 bias=True,
                 with_modulated_dcn=True):
        super().__init__()
        if not with_modulated_dcn:

            pass

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.deformable_groups = deformable_groups
        self.groups = groups

        self.offset_channels = 2 * kh * kw * deformable_groups
        self.mask_channels = kh * kw * deformable_groups
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            self.offset_channels + self.mask_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )

        try:
            self.dcn = TVDeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=False,
                deformable_groups=deformable_groups,
            )
        except TypeError:
            try:
                self.dcn = TVDeformConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                    bias=False,
                    offset_groups=deformable_groups,
                )
            except TypeError:
                self.dcn = TVDeformConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                    bias=False,
                )


        self._init_parameters()

    def _init_parameters(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)
        nn.init.kaiming_normal_(self.dcn.weight, mode='fan_out', nonlinearity='relu')
        if self.dcn.bias is not None:
            nn.init.constant_(self.dcn.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        offset = out[:, :self.offset_channels, :, :]
        mask = out[:, self.offset_channels:, :, :].sigmoid()
        return self.dcn(x, offset, mask)


def dcn_v2_conv(*args, **kwargs):
    return DCN(*args, **kwargs)

