"""
Definitions of the Spectral Normalization discriminator (https://arxiv.org/abs/1802.05957) and a normal discriminator.
"""

from math import floor

import torch
from torch import nn as nn
from torch.nn import functional as F, Linear
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair


def max_singular_value(W, u=None, Ip=1, use_gpu=True):
    """
    power iteration for weight parameter
    borrow from https://github.com/godisboy/SN-GAN
    """

    if u is None:
        if use_gpu:
            u = torch.randn(1, W.size(0)).normal_(0, 1).cuda()
        else:
            u = torch.randn(1, W.size(0)).normal_(0, 1)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, torch.transpose(W.data, 0, 1)), torch.transpose(_u, 0, 1))
    return sigma, _u


def _l2normalize(v, eps=1e-12):
    '''
    borrow from https://github.com/godisboy/SN-GAN
    '''

    return v / (((v**2).sum())**0.5 + eps)


class SNConv2d(conv._ConvNd):
    """A spectral-normalized convolutional layer based on Miyato et al. (https://arxiv.org/abs/1802.05957)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 Ip=1):
        """Constructor

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution
        :param padding: Zero-padding added to both sides of the input
        :param dilation: Spacing between kernel elements
        :param groups: Number of blocked connections from input channels to output channels
        :param bias: If True, adds a learnable bias to the output
        :param Ip: Number of power iterations used to compute max singular value
        """

        self.Ip = Ip
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.u = None
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u, Ip=self.Ip)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SNLinear(Linear):
    """A spectral-normalized linear layer based on Miyato et al. (https://arxiv.org/abs/1802.05957)."""

    def __init__(self, in_features, out_features, bias=True, Ip=1):
        """Constructor

        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param bias: If True, adds a learnable bias to the output
        :param Ip: Number of power iterations used to compute max singular value
        """

        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None
        self.Ip = Ip

    def forward(self, input):
        w_mat = self.weight
        sigma, _u = max_singular_value(w_mat, self.u, Ip=self.Ip)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.linear(input, self.weight, self.bias)


class SNDiscriminator(nn.Module):
    """A discriminator with spectral-normalized convolution and linear layers based on Miyato et al.
    (https://arxiv.org/abs/1802.05957)."""

    def __init__(self, img_size, c_dim, t, df_dim, Ip):
        """Constructor

        :param img_size: The spatial resolution of the video
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param t: The number of frames in a video
        :param df_dim: Controls the number of features in each layer of the discriminator
        :param Ip: The number of power iterations used to compute the maximum singular value
        """

        super(SNDiscriminator, self).__init__()
        h, w = img_size[0], img_size[1]

        conv0 = SNConv2d(c_dim * t, df_dim, 4, stride=2, padding=1, Ip=Ip)
        h = floor((h + 2*1 - 4)/2 + 1)
        w = floor((w + 2*1 - 4)/2 + 1)
        lrelu0 = nn.LeakyReLU(0.2)

        conv1 = SNConv2d(df_dim, df_dim * 2, 4, stride=2, padding=1, Ip=Ip)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        lrelu1 = nn.LeakyReLU(0.2)

        conv2 = SNConv2d(df_dim * 2, df_dim * 4, 4, stride=2, padding=1, Ip=Ip)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        lrelu2 = nn.LeakyReLU(0.2)

        conv3 = SNConv2d(df_dim * 4, df_dim * 8, 4, stride=2, padding=1, Ip=Ip)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        lrelu3 = nn.LeakyReLU(0.2)

        self.D = nn.Sequential(conv0, lrelu0, conv1, lrelu1, conv2, lrelu2, conv3, lrelu3)

        self.num_sn_linear_in_feats = int(h * w * df_dim * 8)

        self.linear = SNLinear(self.num_sn_linear_in_feats, 1, Ip=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        D_output = self.D(input)
        D_output = D_output.view(-1, self.num_sn_linear_in_feats)
        h = self.linear(D_output)
        h_sigmoid = self.sigmoid(h)
        return h_sigmoid, h


class Discriminator(nn.Module):
    """A normal discriminator with batch normalization."""

    def __init__(self, img_size, c_dim, t, df_dim):
        """Constructor

        :param img_size: The spatial resolution of the video
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param t: The number of frames in a video
        :param df_dim: Controls the number of features in each layer of the discriminator
        """

        super(Discriminator, self).__init__()
        h, w = img_size[0], img_size[1]

        conv0 = nn.Conv2d(c_dim * t, df_dim, 4, stride=2, padding=1)
        h = floor((h + 2*1 - 4)/2 + 1)
        w = floor((w + 2*1 - 4)/2 + 1)
        lrelu0 = nn.LeakyReLU(0.2)

        conv1 = nn.Conv2d(df_dim, df_dim * 2, 4, stride=2, padding=1)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        bn1 = nn.BatchNorm2d(df_dim * 2)
        lrelu1 = nn.LeakyReLU(0.2)

        conv2 = nn.Conv2d(df_dim * 2, df_dim * 4, 4, stride=2, padding=1)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        bn2 = nn.BatchNorm2d(df_dim * 4)
        lrelu2 = nn.LeakyReLU(0.2)

        conv3 = nn.Conv2d(df_dim * 4, df_dim * 8, 4, stride=2, padding=1)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        bn3 = nn.BatchNorm2d(df_dim * 8)
        lrelu3 = nn.LeakyReLU(0.2)

        self.D = nn.Sequential(conv0, lrelu0, conv1, bn1, lrelu1, conv2, bn2, lrelu2, conv3, bn3, lrelu3)

        self.num_sn_linear_in_feats = int(h * w * df_dim * 8)

        self.linear = nn.Linear(self.num_sn_linear_in_feats, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        D_output = self.D(input)
        D_output = D_output.view(-1, self.num_sn_linear_in_feats)
        h = self.linear(D_output)
        h_sigmoid = self.sigmoid(h)
        return h_sigmoid, h