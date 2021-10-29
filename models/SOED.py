import torch
from models.unetresnet import UNetResNet, unet18
import torch.nn as nn
import functools
import math
import torch.nn.functional as F
from functools import partial


class FReLU(nn.Module):
   r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
   """
   def __init__(self, in_channels):
       super().__init__()
       self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
       self.bn_frelu = nn.BatchNorm2d(in_channels)

   def forward(self, x):
       x1 = self.conv_frelu(x)
       x1 = self.bn_frelu(x1)
       x = torch.max(x, x1)
       return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



class SOED3_act(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        # polygon extraction
        self.segmentNet = unet18(num_classes=num_classes)

        ## ------------gaussian------------#
        self.segmentNet2 = unet18(num_classes=9, in_ch=in_ch + num_classes-1)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes + 7, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, gaussian=False, seg=False, centerline=False):
        segments = self.segmentNet(x)
        segments_feature = torch.relu(segments[:, 1:, :, :])

        x_gaussian = torch.cat([x, segments_feature], dim=1)
        gaussains = self.segmentNet2(x_gaussian)
        gaussains_feature = torch.relu(gaussains[:, 1:, :, :])

        x_centerline = torch.cat([x, segments_feature, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return segments
        if gaussian:
            return gaussains
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers


class SOED2_act(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        self.segmentNet = unet18(num_classes=num_classes)

        # ------------cropland edge detection------------#
        self.edge_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes - 1, num_filters, norm_layer, num_block=2))
        self.side_edge_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_edge_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_edge_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_edge_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_edge_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _edge_forward(self, x):
        """
        predict road edge
        :param: x, [image tensor, predicted segmentation tensor], [N, C+1, H, W]
        """
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.edge_conv1(x)
        conv2 = self.edge_conv2(self.maxpool(conv1))
        conv3 = self.edge_conv3(self.maxpool(conv2))
        conv4 = self.edge_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_edge_conv1(conv1)
        side_output2 = self.side_edge_conv2(conv2)
        side_output3 = self.side_edge_conv3(conv3)
        side_output4 = self.side_edge_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        fused = self.fuse_edge_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, centerline=False, seg=False):
        segments = self.segmentNet(x)
        segments_feature = torch.relu(segments[:, 1:, :, :])
        x_ = torch.cat([x, segments_feature], dim=1)

        centers = self._edge_forward(x_)

        if seg:
            return segments
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, centers


class SOED2_gaussian_centerline(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        ## ------------gaussian------------#
        self.segmentNet2 = unet18(num_classes=9, in_ch=in_ch)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes - 1, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, seg=False, centerline=False):

        gaussains = self.segmentNet2(x)
        gaussains_feature = torch.relu(gaussains[:, 1:, :, :])

        x_centerline = torch.cat([x, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return gaussains

        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers


class SOED3_V2(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        # polygon extraction
        self.segmentNet = unet18(num_classes=num_classes)

        ## ------------gaussian------------#
        self.segmentNet2 = unet18(num_classes=9, in_ch=in_ch + num_classes-1)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, gaussian=False, seg=False, centerline=False):
        segments = self.segmentNet(x)
        segments_feature = torch.relu(segments[:, 1:, :, :])

        x_gaussian = torch.cat([x, segments_feature], dim=1)
        gaussains = self.segmentNet2(x_gaussian)
        gaussains2 = torch.sum(gaussains[:, 1:, :, :], dim=1) / (gaussains.size()[1] - 1)
        gaussains2 = torch.reshape(gaussains2, [2, 1, gaussains.size()[2], gaussains.size()[3]])
        gaussains_feature = torch.relu(gaussains2)

        x_centerline = torch.cat([x, segments_feature, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return segments
        if gaussian:
            return gaussains
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers


class SOED3_V3(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        # polygon extraction
        self.segmentNet = unet18(num_classes=num_classes)

        ## ------------gaussian------------#
        self.segmentNet2 = unet18(num_classes=9, in_ch=in_ch + num_classes-1)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, gaussian=False, seg=False, centerline=False):
        segments = self.segmentNet(x)
        segments_feature = torch.relu(segments[:, 1:, :, :])

        x_gaussian = torch.cat([x, segments_feature], dim=1)
        gaussains = self.segmentNet2(x_gaussian)
        gaussains2 = gaussains[:, 1:, :, :]
        max_activations, idx = torch.max(gaussains2, dim=1)

        max_activations = torch.reshape(max_activations, [gaussains.size()[0], 1, gaussains.size()[2], gaussains.size()[3]])
        gaussains_feature = torch.relu(max_activations)

        x_centerline = torch.cat([x, segments_feature, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return segments
        if gaussian:
            return gaussains
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers


class SOED3_V4(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        # polygon extraction
        self.segmentNet = unet18(num_classes=num_classes)
        self.SegNorm = nn.BatchNorm2d(2, affine=True)

        ## ------------gaussian------------#
        self.gaussianNet = unet18(num_classes=9, in_ch=in_ch + num_classes-1)
        self.GaussianNorm = nn.BatchNorm2d(1, affine=True)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, gaussian=False, seg=False, centerline=False):
        segments = self.segmentNet(x)
        segments_feature = self.SegNorm(torch.relu(segments[:, 1:, :, :]))

        x_gaussian = torch.cat([x, segments_feature], dim=1)
        gaussains = self.gaussianNet(x_gaussian)
        gaussains2 = gaussains[:, 1:, :, :]
        max_activations, idx = torch.max(gaussains2, dim=1)

        max_activations = torch.reshape(max_activations, [gaussains.size()[0], 1, gaussains.size()[2], gaussains.size()[3]])
        gaussains_feature = self.GaussianNorm(torch.relu(max_activations))

        x_centerline = torch.cat([x, segments_feature, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return segments
        if gaussian:
            return gaussains
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers

class SOED3_V5(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        # polygon extraction
        self.segmentNet = unet18(num_classes=num_classes)
        self.SegNorm = nn.BatchNorm2d(2, affine=True)

        ## ------------gaussian------------#
        self.gaussianNet = unet18(num_classes=9, in_ch=in_ch + num_classes-1)
        self.GaussianNorm = nn.BatchNorm2d(1, affine=True)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + num_classes, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, gaussian=False, seg=False, centerline=False):
        segments = self.segmentNet(x)
        segments_feature = torch.relu(self.SegNorm(segments[:, 1:, :, :]))

        x_gaussian = torch.cat([x, segments_feature], dim=1)
        gaussains = self.gaussianNet(x_gaussian)
        gaussains2 = gaussains[:, 1:, :, :]
        max_activations, idx = torch.max(gaussains2, dim=1)

        max_activations = torch.reshape(max_activations, [gaussains.size()[0], 1, gaussains.size()[2], gaussains.size()[3]])
        gaussains_feature = torch.relu(self.GaussianNorm(max_activations))

        x_centerline = torch.cat([x, segments_feature, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return segments
        if gaussian:
            return gaussains
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers


class SOED3_Multi_class(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, out_nc=1, num_filters=64, norm='batch'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        # polygon extraction
        self.segmentNet = unet18(num_classes=num_classes)

        ## ------------gaussian------------#
        self.segmentNet2 = unet18(num_classes=9, in_ch=in_ch + 2)

        # ------------cropland centerline detection------------#
        self.center_conv1 = nn.Sequential(*self._conv_block(in_ch + 3, num_filters, norm_layer, num_block=2))
        self.side_center_conv1 = nn.Conv2d(num_filters, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv2 = nn.Sequential(*self._conv_block(num_filters, num_filters * 2, norm_layer, num_block=2))
        self.side_center_conv2 = nn.Conv2d(num_filters * 2, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv3 = nn.Sequential(*self._conv_block(num_filters * 2, num_filters * 4, norm_layer, num_block=2))
        self.side_center_conv3 = nn.Conv2d(num_filters * 4, out_nc, kernel_size=1, stride=1, bias=False)

        self.center_conv4 = nn.Sequential(*self._conv_block(num_filters * 4, num_filters * 8, norm_layer,  num_block=2))
        self.side_center_conv4 = nn.Conv2d(num_filters * 8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_center_conv = nn.Conv2d(out_nc * 4, out_nc, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def _conv_block(self, in_nc, out_nc, norm_layer,num_block=2, kernel_size=3,
                    stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)]
            conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _centerline_forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.center_conv1(x)
        conv2 = self.center_conv2(self.maxpool(conv1))
        conv3 = self.center_conv3(self.maxpool(conv2))
        conv4 = self.center_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_center_conv1(conv1)
        side_output2 = self.side_center_conv2(conv2)
        side_output3 = self.side_center_conv3(conv3)
        side_output4 = self.side_center_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
                                     align_corners=True)  # self.up8(side_output4)
        fused = self.fuse_center_conv(torch.cat([
            side_output1,
            side_output2,
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x, gaussian=False, seg=False, centerline=False):
        b, c, w, h = x.size()
        segments = self.segmentNet(x)
        segments_poly = torch.relu(segments[:, 1:-1, :, :])
        segments_line = torch.relu(segments[:, -1, :, :])
        max_segments_activations, _ = torch.max(segments_poly, dim=1)
        max_segments_activations = torch.reshape(max_segments_activations, [b, 1, w, h])
        segments_line = torch.reshape(segments_line, [b, 1, w, h])

        x_gaussian = torch.cat([x, max_segments_activations, segments_line], dim=1)
        gaussains = self.segmentNet2(x_gaussian)
        gaussains2 = gaussains[:, 1:, :, :]
        max_gaussian_activations, _ = torch.max(gaussains2, dim=1)

        max_gaussian_activations = torch.reshape(max_gaussian_activations, [b, 1, w, h])
        gaussains_feature = torch.relu(max_gaussian_activations)

        x_centerline = torch.cat([x, max_segments_activations, segments_line, gaussains_feature], dim=1)
        centers = self._centerline_forward(x_centerline)

        if seg:
            return segments
        if gaussian:
            return gaussains
        if centerline:
            return centers
            # return [edge + segments[:, 2, :, :] for edge in edges]

        return segments, gaussains, centers

if __name__ == '__main__':
    device = 'cpu'
    net = SOED3_V3(num_classes=250).cuda()
    img = torch.randn((2, 3, 256, 256)).cuda()
    # dec = DecoderBlock(512, 256)
    # out = dec(img)
    # print(out.size())
    segments, gaussains, centers = net(img)
    # print(gaussains.size())
    # for hm in centers:
    #     print(hm.size())


    # import numpy as np
    #
    # X = [[-2.1, 2, 3],
    #      [1, -6.1, 5],
    #      [0, 1, 1]]
    #
    # s = np.argmax(np.abs(X), axis=0)
    # # You might need to do this to get X as an ndarray (for example if X is a list)
    # X = np.asarray(X)

    # Or more generally
    # X_argmax = X[s, np.arange(X.shape[1])]
    # print(np.arange(X.shape[1]))
    # print(X_argmax)